"""Legacy threshold-search helpers."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

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

GRID_NX = 256
GRID_NY = 256
GRID_LX = 250.0
GRID_LY = 250.0
SIGMA_SPACE = 5.0
CUTOFF_SIGMA = 3.0
COND_GROWTH_FACTOR = 1e6
CHECK_EVERY = 500
MIN_CHECK_TIME = 50.0
EARLY_STOP_FRAC = 0.33
RNG_SEED = 42
STABILIZATION_PERIODS = 2
MIN_MONITOR_SAMPLES = 10

P_VALUES = [5.0, 10.0, 20.0, 40.0, 60.0, 80.0]
SIGMA_TIME_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
PULSE_SEP_VALUES = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0]


def _build_infra_config(t_max, dt):
    """Build the base config for the legacy search."""
    return Config(
        grid=GridParameters(
            nx=GRID_NX, ny=GRID_NY, lx=GRID_LX, ly=GRID_LY, grid_type="periodic"
        ),
        boundary_condition=BoundaryConditionParameters(absorption="cap"),
        potential=PotentialParameters(potential_type="zero"),
        physics=PhysicsConstants(),
        laser=LaserParameters(
            mode="single",
            laser_type="pulse-gaussian",
            P0=1.0,
            Pmax=1.0,
            x0=0.0,
            y0=0.0,
            sigma_space=SIGMA_SPACE,
            sigma_time=1.0,
            pulse_separation=10.0,
            cutoff_sigma=CUTOFF_SIGMA,
        ),
        reservoir=ReservoirParameters(reservoir_type="double"),
        solver=SolverParameters(total_time=t_max, dt=dt, method="rk4-cuda"),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def _p_time_pure(t, pulse_separation, cutoff_sigma, sigma_time):
    """Return the pulse envelope at time t."""
    n = round(t / pulse_separation)
    dt_val = t - n * pulse_separation
    if abs(dt_val) > cutoff_sigma * sigma_time:
        return 0.0
    return math.exp(-0.5 * (dt_val / sigma_time) ** 2)


class _ReusableInfra:
    """Cache reusable objects for repeated search runs."""
    def __init__(self, t_max, dt):
        """Set up reusable objects for the legacy search."""
        cfg = _build_infra_config(t_max, dt)
        self.xp = compute_engine.xp
        self.grid = create_grid(cfg.grid)
        self.bc = BoundaryCondition(self.grid, cfg.boundary_condition, cfg.physics)
        self.solver = create_solver(cfg, self.grid)
        self.physics = cfg.physics

        potential = create_potential(cfg.potential, self.grid)
        cap = self.bc.before_step_action()
        if self.xp.iscomplexobj(cap) and not self.xp.iscomplexobj(potential):
            potential = potential.astype(self.xp.complex128)
            cap = cap.astype(self.xp.complex128)
        self.potential = potential + cap

        self.n_steps = int(t_max / dt)
        self.dt = dt
        self.t_max = t_max

        rng = self.xp.random.default_rng(RNG_SEED)
        self._init_psi = (
            cfg.physics.init_eps
            * (
                rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
                + 1j * rng.random((self.grid.ny, self.grid.nx), dtype=self.xp.float64)
            )
        ).astype(self.xp.complex128)
        self.N_initial = float(self.xp.sum(self.xp.abs(self._init_psi) ** 2))

    def fresh_state(self):
        """Return a fresh simulation state."""
        state = SimulationState.__new__(SimulationState)
        state.psi = self._init_psi.copy()
        state.t = 0.0
        return state


def evaluate_params(infra, power, sigma_time, pulse_sep):
    """Run one legacy threshold-search point."""
    xp = infra.xp
    grid = infra.grid
    dt = infra.dt

    if CUTOFF_SIGMA * sigma_time >= pulse_sep / 2.0:
        return {"condensed": False, "reason": "pulses_overlap"}

    laser_cfg = LaserParameters(
        mode="single",
        laser_type="pulse-gaussian",
        P0=power,
        Pmax=power,
        x0=0.0,
        y0=0.0,
        sigma_space=SIGMA_SPACE,
        sigma_time=sigma_time,
        pulse_separation=pulse_sep,
        cutoff_sigma=CUTOFF_SIGMA,
    )
    laser = PulseGaussian(laser_cfg, grid.X, grid.Y)
    spatial_profile = laser._P_space(grid.X, grid.Y)
    P_zero = xp.zeros_like(spatial_profile)

    state = infra.fresh_state()
    reservoir = create_reservoir(
        ReservoirParameters(reservoir_type="double"), infra.physics, grid
    )

    t_cond = None
    N_trace = []
    early_stop_step = int(infra.n_steps * EARLY_STOP_FRAC)

    for step in range(infra.n_steps):
        t = step * dt

        amp = laser._amplitude(t)
        pt = _p_time_pure(t, pulse_sep, CUTOFF_SIGMA, sigma_time)
        temporal = float(amp) * pt

        if temporal == 0.0:
            P_total = P_zero
        else:
            P_total = temporal * spatial_profile

        infra.solver.step(infra.potential, P_total, reservoir, infra.bc, state)

        if step > 0 and step % CHECK_EVERY == 0:
            t_now = (step + 1) * dt
            N_total = float(xp.sum(xp.abs(state.psi) ** 2))

            if math.isnan(N_total) or math.isinf(N_total):
                return {"condensed": False, "reason": "diverged"}

            if t_cond is None:
                if (
                    t_now >= MIN_CHECK_TIME
                    and N_total > infra.N_initial * COND_GROWTH_FACTOR
                ):
                    t_cond = t_now
                elif step >= early_stop_step:
                    return {"condensed": False, "reason": "no_condensation"}

            if t_cond is not None:
                N_trace.append((t_now, N_total))

    if t_cond is None:
        return {"condensed": False, "reason": "no_condensation"}

    stabilization_t = t_cond + STABILIZATION_PERIODS * pulse_sep
    monitor_data = [(t, N) for t, N in N_trace if t >= stabilization_t]

    if len(monitor_data) < MIN_MONITOR_SAMPLES:
        return {
            "condensed": True,
            "t_cond": t_cond,
            "modulation": 0.0,
            "reason": "insufficient_monitor_data",
        }

    N_values = np.array([N for _, N in monitor_data])
    N_max = float(np.max(N_values))
    N_min = float(np.min(N_values))
    modulation = (N_max - N_min) / N_max if N_max > 0 else 0.0

    return {
        "condensed": True,
        "t_cond": t_cond,
        "modulation": modulation,
        "N_max": N_max,
        "N_min": N_min,
        "monitor_samples": len(monitor_data),
    }


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_max", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    args = parser.parse_args()

    t_max = args.t_max
    dt = args.dt

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    print("  Building reusable simulation infrastructure...")
    infra = _ReusableInfra(t_max, dt)

    combos = [
        (P, sigma_t, pulse_sep)
        for P in P_VALUES
        for sigma_t in SIGMA_TIME_VALUES
        for pulse_sep in PULSE_SEP_VALUES
        if CUTOFF_SIGMA * sigma_t < pulse_sep / 2.0
    ]

    print(f"  Grid search: {len(combos)} parameter combinations")
    print(f"    P:          {P_VALUES}")
    print(f"    sigma_time: {SIGMA_TIME_VALUES}")
    print(f"    pulse_sep:  {PULSE_SEP_VALUES}")
    print()

    results = []
    for i, (P, sigma_t, pulse_sep) in enumerate(combos):
        label = f"P={P:6.1f}  sigma_t={sigma_t:.1f}  T_sep={pulse_sep:5.1f}"
        print(f"  [{i+1:3d}/{len(combos)}] {label} ... ", end="", flush=True)

        result = evaluate_params(infra, P, sigma_t, pulse_sep)
        result.update({"P": P, "sigma_time": sigma_t, "pulse_separation": pulse_sep})
        results.append(result)

        if result["condensed"]:
            mod = result["modulation"]
            print(f"t_cond={result['t_cond']:7.1f}  modulation={mod:.3f}")
        else:
            print(result["reason"])

    condensed = [r for r in results if r["condensed"] and r.get("modulation", 0) > 0]
    condensed.sort(key=lambda r: r["modulation"], reverse=True)

    print("\n  " + "=" * 70)
    if condensed:
        print("  Top 5 configurations (highest modulation):\n")
        print(
            f"  {'P':>8}  {'sigma_t':>8}  {'T_sep':>8}  {'t_cond':>8}  {'modulation':>10}"
        )
        print(f"  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>8}  {'---':>10}")
        for r in condensed[:5]:
            print(
                f"  {r['P']:8.1f}  {r['sigma_time']:8.1f}  {r['pulse_separation']:8.1f}"
                f"  {r['t_cond']:8.1f}  {r['modulation']:10.4f}"
            )
    else:
        any_cond = [r for r in results if r["condensed"]]
        if any_cond:
            print("  WARNING: All condensed configs have zero modulation.")
            print(
                "           Consider increasing pulse_separation or decreasing power."
            )
        else:
            print("  ERROR: No condensation found in any configuration.")
            print("         The jobs are will not be created")
            return

    best = condensed[0] if condensed else [r for r in results if r["condensed"]][0]

    optimal = {
        "P_optimal": best["P"],
        "t_cond": best["t_cond"],
        "modulation": best["modulation"],
        "t_max": t_max,
        "dt": dt,
        "nx": GRID_NX,
        "ny": GRID_NY,
        "lx": GRID_LX,
        "ly": GRID_LY,
        "sigma_space": SIGMA_SPACE,
        "sigma_time": best["sigma_time"],
        "pulse_separation": best["pulse_separation"],
        "cutoff_sigma": CUTOFF_SIGMA,
        "cond_growth_factor": COND_GROWTH_FACTOR,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    opt_path = os.path.join(script_dir, "optimal_config.json")
    with open(opt_path, "w") as f:
        json.dump(optimal, f, indent=2)

    log_path = os.path.join(script_dir, "grid_search_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n  Best: P={best['P']:.1f}, sigma_t={best['sigma_time']:.1f}, "
        f"T_sep={best['pulse_separation']:.1f}  (modulation={best['modulation']:.4f})"
    )
    print(f"  Saved: {opt_path}")
    print(f"  Full log: {log_path}")


if __name__ == "__main__":
    main()
