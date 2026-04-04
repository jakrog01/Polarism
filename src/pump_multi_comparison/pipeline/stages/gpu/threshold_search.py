"""GPU stage: threshold search.

Scans the (power, sigma_time, pulse_separation) parameter space for the
minimum pump power that drives condensation.  Outputs
``threshold_result.json`` to the run directory.

Invoked by Slurm as:
    python -m pipeline.stages.gpu.threshold_search \\
        --config <run_dir>/config.yaml --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

from pipeline.config.loader import get_threshold_search_cfg, load_config, make_dataclass
from pipeline.manifest.io import atomic_write_json, set_manifest_field
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    ComputeEngineParameters,
    Config,
    GridParameters,
    LaserParameters,
    PhysicsConstants,
    PotentialParameters,
    ReservoirParameters,
    ResultParameters,
    SolverParameters,
)
from polarism.grid.create_grid import create_grid
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver

COND_GROWTH_FACTOR = 1e6
CHECK_EVERY = 500
MIN_CHECK_TIME = 50.0
RNG_SEED = 42


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_config(global_cfg: dict[str, Any], t_max: float, dt: float) -> Config:
    """Build a Config sized for the threshold search (coarser dt, shorter time)."""
    defaults = global_cfg.get("laser_defaults", {})
    return Config(
        grid=make_dataclass(GridParameters, global_cfg.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, global_cfg.get("boundary_condition", {})
        ),
        potential=PotentialParameters(potential_type="zero"),
        physics=make_dataclass(PhysicsConstants, global_cfg.get("physics", {})),
        laser=LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=1.0, Pmax=1.0, x0=0.0, y0=0.0,
            sigma_space=defaults.get("sigma_space", 5.0),
            sigma_time=1.0,
            pulse_separation=10.0,
            cutoff_sigma=defaults.get("cutoff_sigma", 3.0),
        ),
        reservoir=make_dataclass(ReservoirParameters, global_cfg.get("reservoir", {})),
        solver=SolverParameters(total_time=t_max, dt=dt,
                                method=global_cfg.get("solver", {}).get("method", "rk4-cuda")),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


class _SearchInfra:
    """Reusable simulation infrastructure for the threshold scan.

    Grid, solver, boundary conditions and potential are built once and
    shared across all parameter evaluations.  A fresh wavefunction state
    is generated per evaluation via :meth:`fresh_state`.
    """

    def __init__(self, global_cfg: dict[str, Any], t_max: float, dt: float) -> None:
        """Set up reusable objects for the threshold search."""
        cfg = _build_config(global_cfg, t_max, dt)
        self.xp = compute_engine.xp
        self.grid = create_grid(cfg.grid)
        self.bc = BoundaryCondition(self.grid, cfg.boundary_condition, cfg.physics)
        self.solver = create_solver(cfg, self.grid)
        self.physics = cfg.physics
        self.reservoir_kw = global_cfg.get("reservoir", {})

        potential = create_potential(cfg.potential, self.grid)
        cap = self.bc.before_step_action()
        if self.xp.iscomplexobj(cap) and not self.xp.iscomplexobj(potential):
            potential = potential.astype(self.xp.complex128)
            cap = cap.astype(self.xp.complex128)
        self.potential = potential + cap

        self.n_steps = int(t_max / dt)
        self.dt = dt

        rng = self.xp.random.default_rng(RNG_SEED)
        self._init_psi = (
            cfg.physics.init_eps
            * (
                rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
                + 1j * rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
            )
        ).astype(self.xp.complex128)
        self.N_initial = float(self.xp.sum(self.xp.abs(self._init_psi) ** 2))

    def fresh_state(self) -> SimulationState:
        """Return a fresh simulation state."""
        state = SimulationState.__new__(SimulationState)
        state.psi = self._init_psi.copy()
        state.t = 0.0
        return state


def _p_time_pure(t: float, pulse_separation: float,
                 cutoff_sigma: float, sigma_time: float) -> float:
    """Return the pulse envelope at time t."""
    n = round(t / pulse_separation)
    dt_val = t - n * pulse_separation
    if abs(dt_val) > cutoff_sigma * sigma_time:
        return 0.0
    return math.exp(-0.5 * (dt_val / sigma_time) ** 2)


def evaluate_threshold(
    infra: _SearchInfra,
    power: float,
    sigma_time: float,
    pulse_sep: float,
    sigma_space: float,
    cutoff_sigma: float,
) -> dict[str, Any]:
    """Evaluate one parameter combination for condensation.

    Returns
    -------
    dict
        Keys: ``condensed`` (bool), and either ``t_cond`` (float) or
        ``reason`` (str).
    """
    xp = infra.xp
    grid = infra.grid
    dt = infra.dt

    if cutoff_sigma * sigma_time >= pulse_sep / 2.0:
        return {"condensed": False, "reason": "pulses_overlap"}

    laser_cfg = LaserParameters(
        mode="single", laser_type="pulse-gaussian",
        P0=power, Pmax=power, x0=0.0, y0=0.0,
        sigma_space=sigma_space, sigma_time=sigma_time,
        pulse_separation=pulse_sep, cutoff_sigma=cutoff_sigma,
    )
    laser = PulseGaussian(laser_cfg, grid.X, grid.Y)
    spatial_profile = laser._P_space(grid.X, grid.Y)
    P_zero = xp.zeros_like(spatial_profile)

    state = infra.fresh_state()
    reservoir = create_reservoir(
        make_dataclass(ReservoirParameters, infra.reservoir_kw),
        infra.physics, grid,
    )

    for step in range(infra.n_steps):
        t = step * dt
        amp = laser._amplitude(t)
        pt = _p_time_pure(t, pulse_sep, cutoff_sigma, sigma_time)
        temporal = float(amp) * pt
        P_total = P_zero if temporal == 0.0 else temporal * spatial_profile
        infra.solver.step(infra.potential, P_total, reservoir, infra.bc, state)

        if step > 0 and step % CHECK_EVERY == 0:
            t_now = (step + 1) * dt
            N_total = float(xp.sum(xp.abs(state.psi) ** 2))
            if math.isnan(N_total) or math.isinf(N_total):
                return {"condensed": False, "reason": "diverged"}
            if t_now >= MIN_CHECK_TIME and N_total > infra.N_initial * COND_GROWTH_FACTOR:
                return {"condensed": True, "t_cond": t_now}

    return {"condensed": False, "reason": "no_condensation"}


# ── Stage entrypoint ──────────────────────────────────────────────────────────

def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU threshold search stage")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-dir", required=True,
        help="Run directory; threshold_result.json is written here.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    g = cfg["global"]
    ts = get_threshold_search_cfg(cfg)
    defaults = g.get("laser_defaults", {})

    total_time: float = g["solver"]["total_time"]
    cond_frac: float = ts["condensation_fraction"]
    dt_mult: int = ts.get("dt_multiplier", 5)
    base_dt: float = g["solver"]["dt"]
    search_dt = base_dt * dt_mult
    search_t_max = cond_frac * total_time
    max_wall: float = ts["max_runtime_minutes"] * 60

    sigma_space: float = defaults.get("sigma_space", 5.0)
    cutoff_sigma: float = defaults.get("cutoff_sigma", 3.0)
    power_values = sorted(ts["power_values"])
    sigma_time_values: list[float] = ts["sigma_time_values"]
    pulse_sep_values = sorted(ts["pulse_separation_values"], reverse=True)

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    print("=" * 60)
    print(" Threshold Search")
    print("=" * 60)
    print(f"  Run dir        : {args.run_dir}")
    print(f"  t_max (search) : {search_t_max:.1f} ps  ({cond_frac * 100:.0f}% of {total_time:.0f} ps)")
    print(f"  dt (search)    : {search_dt}")
    print(f"  Wall-clock cap : {ts['max_runtime_minutes']} min")
    print(f"  Powers         : {power_values}")
    print(f"  Pulse seps     : {pulse_sep_values}")
    print(f"  sigma_times    : {sigma_time_values}")
    print()

    infra = _SearchInfra(g, search_t_max, search_dt)
    print(f"  Grid {infra.grid.nx}x{infra.grid.ny}, {infra.n_steps} steps\n")

    wall_start = time.time()
    best: dict[str, Any] | None = None
    tested = 0

    for pulse_sep in pulse_sep_values:
        if best is not None:
            break
        for power in power_values:
            if best is not None:
                break
            for sigma_time in sigma_time_values:
                elapsed = time.time() - wall_start
                if elapsed > max_wall:
                    print(f"\n  Wall-clock limit reached ({elapsed:.0f}s)")
                    break
                if cutoff_sigma * sigma_time >= pulse_sep / 2.0:
                    continue

                tested += 1
                label = (f"T_sep={pulse_sep:5.1f}  P={power:6.1f}  "
                         f"sigma_t={sigma_time:.2f}")
                print(f"  [{tested:3d}] {label} ... ", end="", flush=True)

                result = evaluate_threshold(
                    infra, power, sigma_time, pulse_sep, sigma_space, cutoff_sigma
                )

                if result["condensed"]:
                    print(f"CONDENSED at t={result['t_cond']:.1f} ps")
                    best = {
                        "P_threshold": power,
                        "sigma_time": sigma_time,
                        "pulse_separation": pulse_sep,
                        "t_cond": result["t_cond"],
                    }
                    break
                else:
                    print(result["reason"])
            else:
                continue
            break
        else:
            continue
        break

    output: dict[str, Any] = {
        "search_completed": best is not None,
        "combinations_tested": tested,
        "wall_time_seconds": round(time.time() - wall_start, 1),
    }
    if best:
        output.update(best)
        output["nx"] = g["grid"]["nx"]
        output["ny"] = g["grid"]["ny"]
        output["lx"] = g["grid"]["lx"]
        output["ly"] = g["grid"]["ly"]
        output["sigma_space"] = sigma_space
        output["cutoff_sigma"] = cutoff_sigma
    else:
        print("\n  ERROR: no parameter combination achieved condensation!")
        print("  Consider increasing power range or condensation_fraction.")

    out_path = os.path.join(args.run_dir, "threshold_result.json")
    atomic_write_json(out_path, output)
    print(f"\n  Saved: {out_path}")

    if best:
        print(f"  P_threshold   = {best['P_threshold']:.1f}")
        print(f"  sigma_time    = {best['sigma_time']:.2f} ps")
        print(f"  pulse_sep     = {best['pulse_separation']:.1f} ps")
        print(f"  t_cond        = {best['t_cond']:.1f} ps")
        try:
            set_manifest_field(args.run_dir, "threshold_complete", True)
            set_manifest_field(args.run_dir, "threshold_result", {
                "P_threshold": best["P_threshold"],
                "sigma_time": best["sigma_time"],
                "pulse_separation": best["pulse_separation"],
            })
        except Exception as exc:
            print(f"  WARNING: could not update manifest: {exc}", file=sys.stderr)

    sys.exit(0 if best else 1)


if __name__ == "__main__":
    main()
