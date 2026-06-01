"""GPU stage: threshold search.

Scans the (power, sigma_time, pulse_separation) parameter space for the
minimum pump power that drives condensation. ``P_threshold`` is defined
as the lowest value in ``power_values`` for which at least one valid
``(sigma_time, pulse_separation)`` combination produces condensation.

By default the search is anchored to the first scenario in ``config.yaml``:

- laser geometry, delays, power modifiers, finite train counts, and the
  scenario potential are reused as the optimization reference
- the search falls back to the legacy single-laser proxy only when no
  scenario is available

Selection is lexicographic and deterministic:

1. minimize power
2. minimize ``sigma_time`` (shortest pulse)
3. minimize condensation time ``t_cond``
4. minimize ``pulse_separation`` (only relevant when ``pulse_separation_values``
   is used; with ``pulse_separation_formula`` each sigma_time maps to exactly
   one derived separation, so this step is never exercised)

This keeps the threshold search aligned with a "weak but fast" operating
point while still guaranteeing condensation. Outputs
``threshold_result.json`` to the run directory.

Invoked by Slurm as:
    python -m pipeline.stages.gpu.threshold_search \\
        --config <run_dir>/config.yaml --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

from pipeline.config.builder import build_scenario_lasers
from pipeline.config.sweep import (
    _PULSE_LASER_TYPES,
    probe5_assisted_ring_calibration_enabled,
    probe5_assisted_probe_calibration_enabled,
    probe5_probe_calibration_enabled,
    probe5_ring_calibration_enabled,
    resolve_laser_type,
)
from pipeline.config.loader import (
    get_threshold_search_cfg,
    load_config,
    make_dataclass,
    resolve_delay,
)
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
from polarism.init_condition import make_initial_psi
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver

COND_GROWTH_FACTOR = 1e6
COND_PSI_SQ_THRESHOLD = 5e-2
CHECK_EVERY = 500
MIN_CHECK_TIME = 50.0
RNG_SEED = 42

def _build_config(
    global_cfg: dict[str, Any],
    t_max: float,
    dt: float,
    potential_cfg: dict[str, Any] | None = None,
) -> Config:
    """Build a Config sized for the threshold search (coarser dt, shorter time)."""
    defaults = global_cfg.get("laser_defaults", {})
    return Config(
        grid=make_dataclass(GridParameters, global_cfg.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, global_cfg.get("boundary_condition", {})
        ),
        potential=make_dataclass(PotentialParameters, potential_cfg or {}),
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

    def __init__(
        self,
        global_cfg: dict[str, Any],
        t_max: float,
        dt: float,
        potential_cfg: dict[str, Any] | None = None,
    ) -> None:
        """Set up reusable objects for the threshold search."""
        cfg = _build_config(global_cfg, t_max, dt, potential_cfg)
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

        init_seed_cfg = getattr(cfg.physics, "init_seed", None)
        self._init_psi = make_initial_psi(
            self.xp, self.grid.ny, self.grid.nx,
            eps=cfg.physics.init_eps,
            mode=getattr(cfg.physics, "init_mode", "legacy_positive_uniform"),
            dx=self.grid.dx,
            dy=self.grid.dy,
            k_cutoff_um=getattr(cfg.physics, "init_k_cutoff_um", None),
            seed=init_seed_cfg if init_seed_cfg is not None else RNG_SEED,
            cdtype=self.xp.complex128,
            rdtype=self.xp.float64,
        )
        self.N_initial = float(self.xp.sum(self.xp.abs(self._init_psi) ** 2))

    def fresh_state(self) -> SimulationState:
        """Return a fresh simulation state."""
        state = SimulationState.__new__(SimulationState)
        state.psi = self._init_psi.copy()
        state.t = 0.0
        return state


def _run_simulation_kernel(
    infra: _SearchInfra,
    lasers: list,
    spatial_profiles: list,
    psi_sq_threshold: float = COND_PSI_SQ_THRESHOLD,
    min_check_time: float = MIN_CHECK_TIME,
    roi_center: tuple[float, float] | None = None,
    roi_radius: float | None = None,
) -> dict[str, Any]:
    """Run the condensation-detection loop for an arbitrary laser configuration.

    Handles any number of lasers with per-laser delay, finite-train cutoff,
    and individual pulse parameters.  Spatial profiles must be pre-computed
    by the caller via ``laser._P_space``.

    Returns
    -------
    dict
        Keys: ``condensed`` (bool), and either ``t_cond`` (float) or
        ``reason`` (str).
    """
    xp = infra.xp
    dt = infra.dt
    P_zero = xp.zeros_like(spatial_profiles[0])
    roi_mask = None
    if roi_center is not None and roi_radius is not None:
        x0, y0 = roi_center
        roi_mask = (
            (infra.grid.X - float(x0)) ** 2
            + (infra.grid.Y - float(y0)) ** 2
            <= float(roi_radius) ** 2
        )

    state = infra.fresh_state()
    reservoir = create_reservoir(
        make_dataclass(ReservoirParameters, infra.reservoir_kw),
        infra.physics, infra.grid,
    )

    for step in range(infra.n_steps):
        t = step * dt
        P_total = None
        for i, laser in enumerate(lasers):
            t_eff = t - laser.delay
            if t_eff < 0:
                continue
            amp = laser._amplitude(t_eff)
            pt = laser._P_time(t_eff)
            temporal = float(amp) * pt
            if temporal != 0.0:
                contrib = temporal * spatial_profiles[i]
                P_total = contrib if P_total is None else P_total + contrib
        if P_total is None:
            P_total = P_zero

        infra.solver.step(infra.potential, P_total, reservoir, infra.bc, state)

        if step > 0 and step % CHECK_EVERY == 0:
            t_now = (step + 1) * dt
            psi_sq = xp.abs(state.psi) ** 2
            psi_sq_max_current = float(xp.max(psi_sq))
            N_total = float(xp.sum(xp.abs(state.psi) ** 2))
            if math.isnan(N_total) or math.isinf(N_total):
                return {"condensed": False, "reason": "diverged"}
            if roi_mask is not None and t_now >= min_check_time:
                roi_mean = float(xp.mean(psi_sq[roi_mask]))
                if roi_mean > psi_sq_threshold:
                    return {
                        "condensed": True,
                        "t_cond": t_now,
                        "roi_mean_psi_sq": roi_mean,
                    }
                continue
            if (
                t_now >= min_check_time
                and (
                    psi_sq_max_current > psi_sq_threshold
                    or N_total > infra.N_initial * COND_GROWTH_FACTOR
                )
            ):
                return {"condensed": True, "t_cond": t_now}

    return {"condensed": False, "reason": "no_condensation"}


def evaluate_threshold(
    infra: _SearchInfra,
    power: float,
    sigma_time: float,
    pulse_sep: float,
    sigma_space: float,
    cutoff_sigma: float,
    n_pulses: int = 0,
    power_definition: str = "peak_amplitude",
    psi_sq_threshold: float = COND_PSI_SQ_THRESHOLD,
    min_check_time: float = MIN_CHECK_TIME,
) -> dict[str, Any]:
    """Evaluate one parameter combination for condensation (single-laser proxy).

    Returns
    -------
    dict
        Keys: ``condensed`` (bool), and either ``t_cond`` (float) or
        ``reason`` (str).
    """
    if cutoff_sigma * sigma_time >= pulse_sep / 2.0:
        return {"condensed": False, "reason": "pulses_overlap"}

    laser_cfg = LaserParameters(
        mode="single", laser_type="pulse-gaussian",
        P0=power, Pmax=power, x0=0.0, y0=0.0,
        sigma_space=sigma_space, sigma_time=sigma_time,
        pulse_separation=pulse_sep, cutoff_sigma=cutoff_sigma,
        n_pulses=n_pulses,
        power_definition=power_definition,
    )
    laser = PulseGaussian(laser_cfg, infra.grid.X, infra.grid.Y)
    return _run_simulation_kernel(
        infra,
        [laser],
        [laser._spatial_envelope],
        psi_sq_threshold=psi_sq_threshold,
        min_check_time=min_check_time,
    )


def evaluate_threshold_scenario(
    infra: _SearchInfra,
    scenario: dict[str, Any],
    global_cfg: dict[str, Any],
    power: float,
    sigma_time: float,
    pulse_sep: float,
    psi_sq_threshold: float = COND_PSI_SQ_THRESHOLD,
    min_check_time: float = MIN_CHECK_TIME,
    threshold_overrides: dict[str, Any] | None = None,
    roi_center: tuple[float, float] | None = None,
    roi_radius: float | None = None,
) -> dict[str, Any]:
    """Evaluate condensation using the scenario laser geometry.

    Builds all laser instances from the scenario definition — positions,
    staggered delays, finite-train counts, and power modifiers — with
    ``power`` substituted as the reference threshold power ``P``.

    A fresh RNG is initialised from ``RNG_SEED`` on every call so the
    result depends only on the parameter tuple and the fixed seed, not on
    the order in which earlier candidates were evaluated.  This keeps the
    search deterministic for all scenario types, including those that use
    the legacy random-phase path inside ``build_scenario_lasers``.

    The potential baked into ``infra.potential`` is the one constructed at
    ``_SearchInfra`` initialisation time; for a scenario-based search that
    is the actual scenario potential, not a forced zero-potential default.

    Returns
    -------
    dict
        Keys: ``condensed`` (bool), and either ``t_cond`` (float) or
        ``reason`` (str).
    """
    rng = np.random.default_rng(RNG_SEED)
    fake_threshold = {
        "P_threshold": power,
        "sigma_time": sigma_time,
        "pulse_separation": pulse_sep,
    }
    if threshold_overrides:
        fake_threshold.update(threshold_overrides)
    lasers, _ = build_scenario_lasers(scenario, global_cfg, fake_threshold, infra.grid, rng)
    spatial_profiles = [laser._spatial_envelope for laser in lasers]
    return _run_simulation_kernel(
        infra,
        lasers,
        spatial_profiles,
        psi_sq_threshold=psi_sq_threshold,
        min_check_time=min_check_time,
        roi_center=roi_center,
        roi_radius=roi_radius,
    )


def _probe5_ring_calibration_scenario(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build the strict four-spot ring-only calibration scenario."""
    if not probe5_ring_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if (
                float(sweep.get("ring_scale", -1.0)) == 1.0
                and float(sweep.get("probe_energy", -1.0)) == 0.0
            ):
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_ring_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = "P"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = 0.0
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_ring_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
    }
    return scenario


def _probe5_probe_calibration_scenario(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Build the strict center-probe-only calibration scenario."""
    if not probe5_probe_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if float(sweep.get("ring_scale", -1.0)) == 0.0:
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_probe_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = 0.0
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = "P"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_probe_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
    }
    return scenario


def _probe5_assisted_probe_calibration_scenario(
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Build center-probe calibration inside a fixed subthreshold ring trap."""
    if not probe5_assisted_probe_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
    ring_cal = sweep_cfg.get("ring_calibration") or {}
    assisted_cal = sweep_cfg.get("assisted_probe_calibration") or {}
    ring_fraction = float(
        assisted_cal.get("ring_fraction", ring_cal.get("safe_fraction", 0.75))
    )

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = None
    for sc in candidates:
        sweep = sc.get("sweep") or {}
        try:
            if float(sweep.get("ring_scale", -1.0)) == 1.0:
                source = sc
                break
        except (TypeError, ValueError):
            continue
    if source is None and candidates:
        source = candidates[0]
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_assisted_probe_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = f"{ring_fraction:.6g}P_ring"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = "P"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_assisted_probe_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
        "ring_fraction": ring_fraction,
    }
    return scenario


def _probe5_assisted_ring_calibration_scenario(
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Build ring-strength calibration with a fixed subthreshold center probe."""
    if not probe5_assisted_ring_calibration_enabled(cfg):
        return None

    global_cfg = cfg["global"]
    sweep_cfg = global_cfg.get("parameter_sweep", {})
    probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
    assisted_cal = sweep_cfg.get("assisted_ring_calibration") or {}
    probe_fraction = float(assisted_cal.get("probe_fraction", 0.75))

    candidates = [
        sc for sc in cfg.get("scenarios", [])
        if (sc.get("sweep") or {}).get("mode") == "probe5"
    ]
    source = candidates[0] if candidates else None
    if source is None:
        return None

    scenario = copy.deepcopy(source)
    scenario["name"] = "probe5_assisted_ring_calibration"
    for ldef in scenario.get("lasers", [])[:4]:
        ldef["power"] = "P"
    for ldef in scenario.get("lasers", []):
        if str(ldef.get("id", "")) == probe_laser_id:
            ldef["power"] = f"{probe_fraction:.6g}P_probe"
            ldef["n_pulses"] = 1
    scenario["sweep"] = {
        "mode": "probe5_assisted_ring_calibration",
        "source_scenario": source.get("name", "probe5"),
        "square_side": sweep_cfg.get("square_side"),
        "probe_laser_id": probe_laser_id,
        "probe_fraction": probe_fraction,
    }
    return scenario


def _run_search_loop(
    infra: _SearchInfra,
    power_values: list[float],
    sigma_time_values: list[float],
    sigma_space: float,
    cutoff_sigma: float,
    max_wall: float,
    pulse_sep_values: list[float] | None = None,
    pulse_sep_formula: str | None = None,
    n_pulses: int = 0,
    scenario: dict[str, Any] | None = None,
    global_cfg: dict[str, Any] | None = None,
    power_definition: str = "peak_amplitude",
    psi_sq_threshold: float = COND_PSI_SQ_THRESHOLD,
    min_check_time: float = MIN_CHECK_TIME,
    threshold_overrides: dict[str, Any] | None = None,
    roi_center: tuple[float, float] | None = None,
    roi_radius: float | None = None,
) -> tuple[dict[str, Any] | None, int, float]:
    """Search for the minimum pump power that achieves condensation.

    Iterates powers in ascending order. For the first power at which
    condensation is possible, the retained operating point is chosen by
    the lexicographic rule:

    1. smallest ``sigma_time``
    2. smallest ``t_cond``
    3. smallest ``pulse_separation`` (only exercised on the explicit-list
       path; with ``pulse_separation_formula`` each sigma_time maps to
       exactly one derived separation so this step is never reached)

    When ``scenario`` is provided the search uses the full scenario laser
    geometry via :func:`evaluate_threshold_scenario`.  Otherwise it falls
    back to the single-laser proxy via :func:`evaluate_threshold`.

    Returns
    -------
    (best, tested, elapsed_seconds)
        ``best`` is ``None`` when no combination condensed (including
        wall-clock timeout before any result).
    """
    wall_start = time.time()
    best: dict[str, Any] | None = None
    tested = 0
    wall_exceeded = False

    for power in power_values:
        best_at_power: dict[str, Any] | None = None
        for sigma_time in sigma_time_values:
            if pulse_sep_formula is not None:
                pulse_sep_candidates = [
                    resolve_delay(
                        pulse_sep_formula,
                        {
                            "sigma_time": sigma_time,
                            "cutoff_sigma": cutoff_sigma,
                            "pulse_separation": 0.0,
                        },
                    )
                ]
            else:
                pulse_sep_candidates = pulse_sep_values or []

            for pulse_sep in pulse_sep_candidates:
                elapsed = time.time() - wall_start
                if elapsed > max_wall:
                    print(f"\n  Wall-clock limit reached ({elapsed:.0f}s)")
                    wall_exceeded = True
                    break
                if cutoff_sigma * sigma_time >= pulse_sep / 2.0:
                    continue

                tested += 1
                label = (f"P={power:6.1f}  sigma_t={sigma_time:.2f}  "
                         f"T_sep={pulse_sep:5.1f}")
                print(f"  [{tested:3d}] {label} ... ", end="", flush=True)

                if scenario is not None:
                    result = evaluate_threshold_scenario(
                        infra,
                        scenario,
                        global_cfg,
                        power,
                        sigma_time,
                        pulse_sep,
                        psi_sq_threshold=psi_sq_threshold,
                        min_check_time=min_check_time,
                        threshold_overrides=threshold_overrides,
                        roi_center=roi_center,
                        roi_radius=roi_radius,
                    )
                else:
                    result = evaluate_threshold(
                        infra, power, sigma_time, pulse_sep, sigma_space, cutoff_sigma,
                        n_pulses, power_definition,
                        psi_sq_threshold=psi_sq_threshold,
                        min_check_time=min_check_time,
                    )

                if result["condensed"]:
                    print(f"CONDENSED at t={result['t_cond']:.1f} ps")
                    candidate = {
                        "P_threshold": power,
                        "sigma_time": sigma_time,
                        "pulse_separation": pulse_sep,
                        "t_cond": result["t_cond"],
                    }
                    if best_at_power is None or (
                        candidate["t_cond"],
                        candidate["pulse_separation"],
                    ) < (
                        best_at_power["t_cond"],
                        best_at_power["pulse_separation"],
                    ):
                        best_at_power = candidate
                else:
                    print(result["reason"])
            if best_at_power is not None:
                best = best_at_power
            if wall_exceeded or best_at_power is not None:
                break
        if best_at_power is not None or wall_exceeded:
            break

    return best, tested, time.time() - wall_start


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
    ring_calibration_scenario = _probe5_ring_calibration_scenario(cfg)
    probe_calibration_scenario = _probe5_probe_calibration_scenario(cfg)
    assisted_probe_calibration_scenario = _probe5_assisted_probe_calibration_scenario(cfg)
    assisted_ring_calibration_scenario = _probe5_assisted_ring_calibration_scenario(cfg)
    first_scenario: dict[str, Any] | None = (
        ring_calibration_scenario or (cfg.get("scenarios") or [None])[0]
    )

    total_time: float = g["solver"]["total_time"]
    cond_frac: float = ts["condensation_fraction"]
    dt_mult: int = ts.get("dt_multiplier", 5)
    base_dt: float = g["solver"]["dt"]
    search_dt = base_dt * dt_mult
    search_t_max = cond_frac * total_time
    max_wall: float = ts["max_runtime_minutes"] * 60

    sigma_space: float = defaults.get("sigma_space", 5.0)
    cutoff_sigma: float = defaults.get("cutoff_sigma", 3.0)
    power_definition: str = defaults.get("power_definition", "peak_amplitude")
    power_values = sorted(ts["power_values"])
    sigma_time_values = sorted(ts["sigma_time_values"])
    psi_sq_threshold = float(ts.get("condensation_psi_sq_threshold", COND_PSI_SQ_THRESHOLD))
    min_check_time = float(ts.get("min_check_time", MIN_CHECK_TIME))
    pulse_sep_formula: str | None = ts.get("pulse_separation_formula")
    pulse_sep_values = sorted(ts.get("pulse_separation_values", []))
    n_pulses: int = int(ts.get("n_pulses", 0))

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    print("=" * 60)
    print(" Threshold Search")
    print("=" * 60)
    print(f"  Run dir        : {args.run_dir}")
    print(f"  t_max (search) : {search_t_max:.1f} ps  ({cond_frac * 100:.0f}% of {total_time:.0f} ps)")
    print(f"  dt (search)    : {search_dt}")
    print(f"  Wall-clock cap : {ts['max_runtime_minutes']} min")
    print(f"  Psi² threshold : {psi_sq_threshold:g}")
    print(f"  Min check time : {min_check_time:g} ps")
    print(f"  Powers         : {power_values}")
    if pulse_sep_formula is not None:
        print(f"  Pulse sep rule : {pulse_sep_formula}")
    else:
        print(f"  Pulse seps     : {pulse_sep_values}")
    print(f"  sigma_times    : {sigma_time_values}")
    if n_pulses > 0:
        print(f"  Max pulses     : {n_pulses}")
    non_pulse_types: list[str] = []
    default_laser_type = defaults.get("laser_type", "pulse-gaussian")
    if default_laser_type not in _PULSE_LASER_TYPES:
        non_pulse_types.append(default_laser_type)
    if first_scenario is not None:
        for ldef in first_scenario.get("lasers", []):
            lt = resolve_laser_type(ldef, defaults)
            if lt not in _PULSE_LASER_TYPES and lt not in non_pulse_types:
                non_pulse_types.append(lt)
    if non_pulse_types:
        print(
            f"ERROR: threshold_search does not support non-pulse laser types "
            f"(found: {non_pulse_types}). "
            "Use parameter_sweep instead — continuous-wave condensation threshold is "
            "a critical-power sweep, not a pulse-parameter search.",
            file=sys.stderr,
        )
        sys.exit(1)
    if first_scenario is not None:
        if (
            ring_calibration_scenario is not None
            and assisted_ring_calibration_scenario is not None
        ):
            print(
                "  Search kernel  : probe5 strict ring-only + probe-only "
                "+ assisted-ring calibration"
            )
        elif (
            ring_calibration_scenario is not None
            and assisted_probe_calibration_scenario is not None
        ):
            print("  Search kernel  : probe5 assisted-probe calibration")
        elif ring_calibration_scenario is not None and probe_calibration_scenario is not None:
            print("  Search kernel  : probe5 strict ring-only + probe-only calibration")
        elif ring_calibration_scenario is not None:
            print("  Search kernel  : probe5 strict ring-only calibration")
        else:
            print(f"  Search kernel  : multi-laser (scenario '{first_scenario.get('name', 'unnamed')}')")
    else:
        print(f"  Search kernel  : single-laser proxy (no scenarios defined)")
    print()

    scenario_potential_cfg = (
        first_scenario.get("potential") if first_scenario is not None else None
    )
    infra = _SearchInfra(g, search_t_max, search_dt, potential_cfg=scenario_potential_cfg)
    print(f"  Grid {infra.grid.nx}x{infra.grid.ny}, {infra.n_steps} steps\n")

    if ring_calibration_scenario is not None:
        print("  Ring-only calibration:")
    best, tested, wall_time = _run_search_loop(
        infra,
        power_values,
        sigma_time_values,
        sigma_space,
        cutoff_sigma,
        max_wall,
        pulse_sep_values=pulse_sep_values,
        pulse_sep_formula=pulse_sep_formula,
        n_pulses=n_pulses,
        scenario=first_scenario,
        global_cfg=g,
        power_definition=power_definition,
        psi_sq_threshold=psi_sq_threshold,
        min_check_time=min_check_time,
    )

    probe_best: dict[str, Any] | None = None
    probe_tested = 0
    probe_wall_time = 0.0
    if probe_calibration_scenario is not None and best is not None:
        print("\n  Probe-only calibration:")
        probe_best, probe_tested, probe_wall_time = _run_search_loop(
            infra,
            power_values,
            sigma_time_values,
            sigma_space,
            cutoff_sigma,
            max_wall,
            pulse_sep_values=pulse_sep_values,
            pulse_sep_formula=pulse_sep_formula,
            n_pulses=1,
            scenario=probe_calibration_scenario,
            global_cfg=g,
            power_definition=power_definition,
            psi_sq_threshold=psi_sq_threshold,
            min_check_time=min_check_time,
        )

    assisted_probe_best: dict[str, Any] | None = None
    assisted_probe_tested = 0
    assisted_probe_wall_time = 0.0
    if assisted_probe_calibration_scenario is not None and best is not None:
        print("\n  Assisted-probe calibration:")
        assisted_probe_best, assisted_probe_tested, assisted_probe_wall_time = _run_search_loop(
            infra,
            power_values,
            sigma_time_values,
            sigma_space,
            cutoff_sigma,
            max_wall,
            pulse_sep_values=pulse_sep_values,
            pulse_sep_formula=pulse_sep_formula,
            n_pulses=1,
            scenario=assisted_probe_calibration_scenario,
            global_cfg=g,
            power_definition=power_definition,
            psi_sq_threshold=psi_sq_threshold,
            min_check_time=min_check_time,
            threshold_overrides={"P_ring_threshold": best["P_threshold"]},
        )

    assisted_ring_best: dict[str, Any] | None = None
    assisted_ring_tested = 0
    assisted_ring_wall_time = 0.0
    if (
        assisted_ring_calibration_scenario is not None
        and best is not None
        and probe_best is not None
    ):
        print("\n  Assisted-ring calibration:")
        sweep_cfg = g.get("parameter_sweep", {})
        probe_laser_id = str(sweep_cfg.get("probe_laser_id", "probe"))
        probe_delay = 0.0
        for ldef in assisted_ring_calibration_scenario.get("lasers", []):
            if str(ldef.get("id", "")) == probe_laser_id:
                probe_delay = float(ldef.get("delay", 0.0))
                break
        assisted_ring_best, assisted_ring_tested, assisted_ring_wall_time = _run_search_loop(
            infra,
            power_values,
            sigma_time_values,
            sigma_space,
            cutoff_sigma,
            max_wall,
            pulse_sep_values=pulse_sep_values,
            pulse_sep_formula=pulse_sep_formula,
            n_pulses=n_pulses,
            scenario=assisted_ring_calibration_scenario,
            global_cfg=g,
            power_definition=power_definition,
            psi_sq_threshold=psi_sq_threshold,
            min_check_time=max(min_check_time, probe_delay),
            threshold_overrides={"P_probe_threshold": probe_best["P_threshold"]},
            roi_center=(0.0, 0.0),
            roi_radius=4.0 * float(sigma_space),
        )

    guard_failure: str | None = None
    if assisted_ring_best is not None and best is not None:
        sweep_cfg = g.get("parameter_sweep", {})
        ring_ref = str(sweep_cfg.get("ring_multiplier_reference", "P_ring"))
        ring_ref_norm = ring_ref.lower().replace("_", "")
        if ring_ref_norm in {"passistedring", "pringassisted"}:
            multipliers = [float(v) for v in sweep_cfg.get("ring_multiplier_values", [1.0])]
            max_multiplier = max(multipliers) if multipliers else 1.0
            assisted_cal = sweep_cfg.get("assisted_ring_calibration") or {}
            guard_fraction = float(assisted_cal.get("ring_guard_fraction", 0.95))
            max_final_ring = max_multiplier * assisted_ring_best["P_threshold"]
            max_safe_ring = guard_fraction * best["P_threshold"]
            if max_final_ring >= max_safe_ring:
                guard_failure = (
                    "Final ring sweep would reach "
                    f"{max_final_ring:.1f}, but guard allows only "
                    f"{max_safe_ring:.1f} = {guard_fraction:.2f}*P_ring. "
                    "Increase the fixed probe_fraction or reduce the max ring multiplier."
                )

    search_completed = best is not None and (
        probe_calibration_scenario is None or probe_best is not None
    ) and (
        assisted_probe_calibration_scenario is None
        or assisted_probe_best is not None
    ) and (
        assisted_ring_calibration_scenario is None
        or assisted_ring_best is not None
    ) and (
        guard_failure is None
    )
    output: dict[str, Any] = {
        "search_completed": search_completed,
        "combinations_tested": (
            tested + probe_tested + assisted_probe_tested + assisted_ring_tested
        ),
        "wall_time_seconds": round(
            wall_time
            + probe_wall_time
            + assisted_probe_wall_time
            + assisted_ring_wall_time,
            1,
        ),
    }
    if guard_failure is not None:
        output["guard_failure"] = guard_failure
    if best is not None:
        output["P_ring_threshold"] = best["P_threshold"]
    if probe_best is not None:
        output["P_probe_threshold"] = probe_best["P_threshold"]
    if assisted_ring_best is not None:
        output["P_assisted_ring_threshold"] = assisted_ring_best["P_threshold"]
        output["P_ring_assisted_threshold"] = assisted_ring_best["P_threshold"]
    if assisted_probe_best is not None:
        output["P_assisted_probe_threshold"] = assisted_probe_best["P_threshold"]
        output["P_probe_assisted_threshold"] = assisted_probe_best["P_threshold"]
    if search_completed and best:
        output.update(best)
        if ring_calibration_scenario is not None:
            output["P_ring_threshold"] = best["P_threshold"]
            output["ring_sigma_time"] = best["sigma_time"]
            output["ring_pulse_separation"] = best["pulse_separation"]
            output["ring_t_cond"] = best["t_cond"]
            output["mode"] = "calibrated_parameter_sweep"
            if probe_best is not None:
                output["P_probe_threshold"] = probe_best["P_threshold"]
                output["probe_sigma_time"] = probe_best["sigma_time"]
                output["probe_pulse_separation"] = probe_best["pulse_separation"]
                output["probe_t_cond"] = probe_best["t_cond"]
            if assisted_probe_best is not None:
                output["P_assisted_probe_threshold"] = assisted_probe_best["P_threshold"]
                output["P_probe_assisted_threshold"] = assisted_probe_best["P_threshold"]
                output["assisted_probe_sigma_time"] = assisted_probe_best["sigma_time"]
                output["assisted_probe_pulse_separation"] = assisted_probe_best["pulse_separation"]
                output["assisted_probe_t_cond"] = assisted_probe_best["t_cond"]
            if assisted_ring_best is not None:
                output["P_assisted_ring_threshold"] = assisted_ring_best["P_threshold"]
                output["P_ring_assisted_threshold"] = assisted_ring_best["P_threshold"]
                output["assisted_ring_sigma_time"] = assisted_ring_best["sigma_time"]
                output["assisted_ring_pulse_separation"] = assisted_ring_best["pulse_separation"]
                output["assisted_ring_t_cond"] = assisted_ring_best["t_cond"]
            ring_safe_fraction = float(
                g.get("parameter_sweep", {})
                .get("ring_calibration", {})
                .get("safe_fraction", 1.0)
            )
            output["calibration"] = {
                "mode": (
                    "probe5_ring_probe_and_assisted_probe"
                    if assisted_probe_best is not None
                    else (
                        "probe5_ring_probe_and_assisted_ring"
                        if assisted_ring_best is not None
                        else (
                            "probe5_ring_and_probe"
                            if probe_best is not None else "probe5_ring_only"
                        )
                    )
                ),
                "condensation_psi_sq_threshold": psi_sq_threshold,
                "min_check_time": min_check_time,
                "safe_fraction": ring_safe_fraction,
                "ring_safe_fraction": ring_safe_fraction,
                "probe_calibration_enabled": probe_best is not None,
                "assisted_probe_calibration_enabled": assisted_probe_best is not None,
                "assisted_ring_calibration_enabled": assisted_ring_best is not None,
                "assisted_probe_ring_fraction": (
                    (assisted_probe_calibration_scenario.get("sweep") or {}).get(
                        "ring_fraction"
                    )
                    if assisted_probe_calibration_scenario is not None
                    else None
                ),
                "assisted_ring_probe_fraction": (
                    (assisted_ring_calibration_scenario.get("sweep") or {}).get(
                        "probe_fraction"
                    )
                    if assisted_ring_calibration_scenario is not None
                    else None
                ),
                "probe_energy_values": (
                    g.get("parameter_sweep", {}).get("probe_energy_values", [])
                ),
            }
        output["nx"] = g["grid"]["nx"]
        output["ny"] = g["grid"]["ny"]
        output["lx"] = g["grid"]["lx"]
        output["ly"] = g["grid"]["ly"]
        output["sigma_space"] = sigma_space
        output["cutoff_sigma"] = cutoff_sigma
        output["power_definition"] = power_definition
    else:
        if guard_failure is not None:
            print("\n  ERROR: threshold guard failed after successful calibration!")
        else:
            print("\n  ERROR: no parameter combination achieved condensation!")
        if best is None:
            print("  Ring/reference search did not find a threshold.")
        if probe_calibration_scenario is not None and probe_best is None:
            print("  Probe-only search did not find a threshold.")
        if (
            assisted_probe_calibration_scenario is not None
            and assisted_probe_best is None
        ):
            print("  Assisted-probe search did not find a threshold.")
        if assisted_ring_calibration_scenario is not None and assisted_ring_best is None:
            print("  Assisted-ring search did not find a threshold.")
        if guard_failure is not None:
            print(f"  Guard failed: {guard_failure}")
            print(
                "  Reduce the final ring multiplier range or increase the fixed "
                "probe_fraction; do not treat this as a missing-condensation case."
            )
        else:
            print("  Consider increasing power range or condensation_fraction.")

    out_path = os.path.join(args.run_dir, "threshold_result.json")
    atomic_write_json(out_path, output)
    print(f"\n  Saved: {out_path}")

    if search_completed and best:
        if ring_calibration_scenario is not None:
            print(f"  P_ring_threshold  = {best['P_threshold']:.1f}")
        else:
            print(f"  P_threshold       = {best['P_threshold']:.1f}")
        print(f"  sigma_time        = {best['sigma_time']:.2f} ps")
        print(f"  pulse_sep         = {best['pulse_separation']:.1f} ps")
        print(f"  t_cond            = {best['t_cond']:.1f} ps")
        if probe_best is not None:
            print(f"  P_probe_threshold = {probe_best['P_threshold']:.1f}")
            print(f"  probe_t_cond      = {probe_best['t_cond']:.1f} ps")
        if assisted_probe_best is not None:
            print(
                f"  P_assisted_probe_threshold = "
                f"{assisted_probe_best['P_threshold']:.1f}"
            )
            print(
                f"  assisted_probe_t_cond      = "
                f"{assisted_probe_best['t_cond']:.1f} ps"
            )
        if assisted_ring_best is not None:
            print(
                f"  P_assisted_ring_threshold  = "
                f"{assisted_ring_best['P_threshold']:.1f}"
            )
            print(
                f"  assisted_ring_t_cond       = "
                f"{assisted_ring_best['t_cond']:.1f} ps"
            )
        try:
            set_manifest_field(args.run_dir, "threshold_complete", True)
            manifest_threshold = {
                "P_threshold": best["P_threshold"],
                "sigma_time": best["sigma_time"],
                "pulse_separation": best["pulse_separation"],
            }
            if ring_calibration_scenario is not None:
                manifest_threshold["P_ring_threshold"] = best["P_threshold"]
            if probe_best is not None:
                manifest_threshold["P_probe_threshold"] = probe_best["P_threshold"]
            if assisted_probe_best is not None:
                manifest_threshold["P_assisted_probe_threshold"] = (
                    assisted_probe_best["P_threshold"]
                )
            if assisted_ring_best is not None:
                manifest_threshold["P_assisted_ring_threshold"] = (
                    assisted_ring_best["P_threshold"]
                )
            set_manifest_field(args.run_dir, "threshold_result", manifest_threshold)
        except Exception as exc:
            print(f"  WARNING: could not update manifest: {exc}", file=sys.stderr)

    sys.exit(0 if search_completed else 1)


if __name__ == "__main__":
    main()
