"""GPU Stage 2: search for the Gaussian dot spatial size that best matches the target response.

Algorithm
---------
For each candidate *sigma_space* from ``fit.sigma_space_values``:

1. Build a 2-D simulation config with that spatial size.
2. For each amplitude in a subsampled sweep, run a short simulation and
   extract the chosen observable (``psi_sq_max``, ``integrated_psi_sq``, or
   ``center_psi_sq``).
3. Compare the resulting amplitude-response curve to the 1-D ODE target from
   ``target_response.json`` using RMSE on normalised curves.

The candidate with the lowest RMSE score is selected.

Output: ``fit_result.json`` in the run directory.

Invoked as::

    python -m dot_response_fit.stages.gpu.fit_dot_size --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

from dot_response_fit.config.loader import (
    build_amplitude_array,
    build_t_eval,
    extract_fit_cfg,
    extract_time_response_cfg,
    load_config,
    make_dataclass,
)
from dot_response_fit.config.builder import build_scenario_config, build_scenario_lasers
from dot_response_fit.manifest.io import (
    atomic_write_json,
    load_scenario_index,
    set_manifest_field,
)
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    ComputeEngineParameters,
    PhysicsConstants,
    ReservoirParameters,
    SolverParameters,
)
from polarism.grid.create_grid import create_grid
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver
from polarism.response.sweep import normalized_integral_curve, rmse_score

RNG_SEED = 42
_MAX_FIT_AMPLITUDES = 10


def _run_single_amplitude_2d(
    infra: "_FitInfra",
    amp: float,
    sigma_time: float,
    pulse_sep: float,
    center: float,
    width_fwhm: float,
    observable: str,
) -> float:
    """Run one short 2-D simulation for a single pump amplitude.

    The pump is modelled as a train with the configured sigma_time and
    pulse_separation.  The simulation runs for just long enough to capture
    the pulse response.

    Returns
    -------
    float
        Value of *observable* extracted from the final recorded state.
    """
    xp = infra.xp
    dt = infra.dt
    n_steps = infra.n_steps

    spatial_profile = infra.spatial_profile
    P_zero = xp.zeros_like(spatial_profile)

    state = infra.fresh_state()
    reservoir = create_reservoir(
        make_dataclass(ReservoirParameters, infra.reservoir_kw),
        infra.physics,
        infra.grid,
    )

    sigma = width_fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    cutoff = infra.cutoff_sigma * sigma_time

    last_psi_sq_max = 0.0
    integrated_psi_sq = 0.0
    prev_t = 0.0

    for step in range(n_steps):
        t = step * dt
        pt = amp * math.exp(-0.5 * ((t - center) / sigma) ** 2) if sigma > 0 else 0.0
        P_total = pt * spatial_profile if abs(pt) > 1e-12 else P_zero

        infra.solver.step(infra.potential, P_total, reservoir, infra.bc, state)

        if step % 50 == 0:
            psi_sq = xp.abs(state.psi) ** 2
            val = float(xp.max(psi_sq))
            if observable in ("psi_sq_max", "integrated_psi_sq"):
                if step > 0:
                    integrated_psi_sq += val * (t - prev_t)
                last_psi_sq_max = val
                prev_t = t

    if observable == "psi_sq_max":
        return last_psi_sq_max
    if observable == "integrated_psi_sq":
        return integrated_psi_sq
    if observable == "center_psi_sq":
        cy, cx = infra.grid.ny // 2, infra.grid.nx // 2
        psi_sq = xp.abs(state.psi) ** 2
        return float(psi_sq[cy, cx])
    return last_psi_sq_max


class _FitInfra:
    """Shared simulation infrastructure for the fit search.

    Grid, solver, BC, and potential are built once; only the wavefunction
    state and reservoir are refreshed per evaluation.
    """

    def __init__(
        self,
        global_cfg: dict[str, Any],
        scenario: dict[str, Any],
        sigma_space: float,
        sigma_time: float,
        pulse_sep: float,
        t_response: float,
        dt: float,
    ) -> None:
        """Build the shared infra for one sigma_space candidate."""
        from dot_response_fit.config.builder import build_scenario_config

        cfg = build_scenario_config(
            global_cfg, scenario, sigma_space, sigma_time, pulse_sep
        )
        cfg.solver.total_time = t_response
        cfg.solver.dt = dt

        self.xp = compute_engine.xp
        self.grid = create_grid(cfg.grid)
        self.bc = BoundaryCondition(self.grid, cfg.boundary_condition, cfg.physics)
        self.physics = cfg.physics
        self.reservoir_kw = global_cfg.get("reservoir", {})
        self.cutoff_sigma = float(global_cfg.get("laser_defaults", {}).get("cutoff_sigma", 3.0))

        potential = create_potential(cfg.potential, self.grid)
        cap = self.bc.before_step_action()
        if self.xp.iscomplexobj(cap) and not self.xp.iscomplexobj(potential):
            potential = potential.astype(self.xp.complex128)
            cap = cap.astype(self.xp.complex128)
        self.potential = potential + cap

        self.solver = create_solver(cfg, self.grid)
        self.dt = dt
        self.n_steps = int(t_response / dt)

        rng = self.xp.random.default_rng(RNG_SEED)
        self._init_psi = (
            cfg.physics.init_eps
            * (
                rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
                + 1j * rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
            )
        ).astype(self.xp.complex128)

        laser = build_scenario_lasers(
            scenario, global_cfg, sigma_space, sigma_time, pulse_sep, self.grid
        )[0]
        self.spatial_profile = laser._P_space(self.grid.X, self.grid.Y)

    def fresh_state(self) -> SimulationState:
        """Return a freshly reset simulation state."""
        state = SimulationState.__new__(SimulationState)
        state.psi = self._init_psi.copy()
        state.t = 0.0
        return state


def _evaluate_sigma_space(
    sigma_space: float,
    global_cfg: dict[str, Any],
    scenario: dict[str, Any],
    fit_amplitudes: np.ndarray,
    target_integrals: list[float],
    tr_cfg: dict[str, Any],
    fit_cfg: dict[str, Any],
    observable: str,
    search_dt: float,
) -> dict[str, Any]:
    """Evaluate one sigma_space candidate; return a result dict."""
    sigma_time = float(global_cfg.get("laser_defaults", {}).get("sigma_time", 0.1))
    pulse_sep = float(global_cfg.get("laser_defaults", {}).get("pulse_separation", 1.0))
    pulse_cfg = tr_cfg["pulse"]
    center = float(pulse_cfg["center"])
    width_fwhm = float(pulse_cfg["width_fwhm"])

    t_response = float(tr_cfg["t_end"]) * 0.5
    infra = _FitInfra(
        global_cfg, scenario, sigma_space, sigma_time, pulse_sep,
        t_response, search_dt,
    )

    candidate_observables: list[float] = []
    for amp in fit_amplitudes:
        val = _run_single_amplitude_2d(
            infra, float(amp), sigma_time, pulse_sep,
            center, width_fwhm, observable,
        )
        candidate_observables.append(val)

    score = rmse_score(target_integrals, candidate_observables)

    return {
        "sigma_space": float(sigma_space),
        "score": score,
        "observable_values": candidate_observables,
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU Stage 2: fit Gaussian dot size")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    target_path = os.path.join(run_dir, "target_response.json")
    for path in (config_path, target_path):
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(config_path)
    global_cfg = cfg["global"]
    fit_cfg = extract_fit_cfg(cfg)
    tr_cfg = extract_time_response_cfg(cfg)

    with open(target_path) as f:
        target = json.load(f)

    target_amplitudes = np.asarray(target["amplitudes"])
    target_integrals = target["nc_integrals"]

    sigma_space_values: list[float] = [float(v) for v in fit_cfg["sigma_space_values"]]
    observable: str = fit_cfg.get("observable", "psi_sq_max")
    max_runtime_minutes: float = float(fit_cfg.get("max_runtime_minutes", 120))
    max_wall = max_runtime_minutes * 60.0

    n_fit = min(_MAX_FIT_AMPLITUDES, len(target_amplitudes))
    idx = np.linspace(0, len(target_amplitudes) - 1, n_fit, dtype=int)
    fit_amplitudes = target_amplitudes[idx]
    fit_target_integrals = [target_integrals[i] for i in idx]

    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        print("ERROR: no scenarios defined in config", file=sys.stderr)
        sys.exit(1)
    scenario = scenarios[0]

    search_dt = float(global_cfg.get("solver", {}).get("dt", 0.001)) * 5

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    print("=" * 60)
    print(" GPU Stage 2: Fit Gaussian Dot Size")
    print("=" * 60)
    print(f"  Run dir         : {run_dir}")
    print(f"  Observable      : {observable}")
    print(f"  sigma_space candidates: {sigma_space_values}")
    print(f"  Fit amplitudes  : {n_fit} subsampled from {len(target_amplitudes)}")
    print(f"  Wall-clock cap  : {max_runtime_minutes} min")
    print()

    wall_start = time.time()
    all_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for sigma_space in sigma_space_values:
        elapsed = time.time() - wall_start
        if elapsed > max_wall:
            print(f"\n  Wall-clock limit reached ({elapsed:.0f}s)")
            break

        print(f"  sigma_space={sigma_space:.2f} μm ... ", end="", flush=True)
        try:
            res = _evaluate_sigma_space(
                sigma_space, global_cfg, scenario,
                fit_amplitudes, fit_target_integrals,
                tr_cfg, fit_cfg, observable, search_dt,
            )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            res = {"sigma_space": float(sigma_space), "score": float("inf"), "observable_values": []}

        all_results.append(res)
        print(f"score={res['score']:.6f}")

        if best is None or res["score"] < best["score"]:
            best = res

    output: dict[str, Any] = {
        "search_completed": best is not None,
        "best_sigma_space": best["sigma_space"] if best else None,
        "best_score": best["score"] if best else None,
        "observable": observable,
        "target_path": target_path,
        "fit_amplitude_indices": idx.tolist(),
        "all_results": all_results,
        "wall_time_seconds": round(time.time() - wall_start, 1),
    }

    out_path = os.path.join(run_dir, "fit_result.json")
    atomic_write_json(out_path, output)
    print(f"\n  Saved: {out_path}")

    if best:
        print(f"  Best sigma_space: {best['sigma_space']:.2f} μm  (score={best['score']:.6f})")
        try:
            set_manifest_field(run_dir, "fit_complete", True)
            set_manifest_field(run_dir, "fit_result", {
                "best_sigma_space": best["sigma_space"],
                "best_score": best["score"],
            })
        except Exception as exc:
            print(f"  WARNING: could not update manifest: {exc}", file=sys.stderr)

    sys.exit(0 if best else 1)


if __name__ == "__main__":
    main()
