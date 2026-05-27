"""Pure simulation kernel for a single power-sweep point.

This module contains only physics logic — no file I/O, no Slurm, no
config-file paths.  The caller supplies a fully parsed config dict and
a target power value; this module returns a structured result.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.grid.create_grid import create_grid
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.init_condition import make_initial_psi
from polarism.simulation_state import SimulationState
import polarism.solver  # noqa: F401 - populate solver registry
from polarism.solver.create_solver import create_solver

from threshold_finder.config.builder import build_power_config

RNG_SEED = 42


@dataclass
class PowerResult:
    """Scalar results for one power-sweep point."""

    power_index: int
    P: float
    status: str
    psi_sq_max: float
    t_psi_sq_max: float
    n_steps_total: int
    last_step_completed: int
    diverged_at_step: int | None
    diverged_at_t: float | None
    wall_time_seconds: float
    scalar_times: list[float] = field(default_factory=list)
    scalar_psi_sq_max: list[float] = field(default_factory=list)


def run_power_point(
    power_index: int,
    P: float,
    cfg: dict[str, Any],
    scalar_check_every: int = 50,
    early_stop_on_divergence: bool = True,
    save_trace: bool = True,
) -> PowerResult:
    """Simulate polariton dynamics at pump power *P* and return scalar results.

    Parameters
    ----------
    power_index : int
        Index of this power value in the sweep array (for identification).
    P : float
        Pump power for this evaluation point.
    cfg : dict
        Full parsed config (output of
        :func:`~threshold_finder.config.loader.load_config`).
    scalar_check_every : int
        Record psi_sq_max every this many simulation steps.
    early_stop_on_divergence : bool
        Terminate the loop immediately on NaN/Inf detection.
    save_trace : bool
        Accumulate the full per-step scalar trace in the result.

    Returns
    -------
    PowerResult
        Scalar results; ``status`` is ``'ok'`` or ``'diverged'``.
    """
    wall_start = time.monotonic()

    sim_cfg = build_power_config(cfg, P)
    xp = compute_engine.xp

    grid = create_grid(sim_cfg.grid)
    bc = BoundaryCondition(grid, sim_cfg.boundary_condition, sim_cfg.physics)
    potential = create_potential(sim_cfg.potential, grid)
    reservoir = create_reservoir(sim_cfg.reservoir, sim_cfg.physics, grid)

    state = SimulationState.__new__(SimulationState)
    state.psi = make_initial_psi(
        xp,
        grid.ny,
        grid.nx,
        eps=sim_cfg.physics.init_eps,
        mode=getattr(sim_cfg.physics, "init_mode", "legacy_positive_uniform"),
        dx=grid.dx,
        dy=grid.dy,
        k_cutoff_um=getattr(sim_cfg.physics, "init_k_cutoff_um", None),
        seed=getattr(sim_cfg.physics, "init_seed", None) or RNG_SEED,
        cdtype=xp.complex128,
        rdtype=xp.float64,
    )
    state.t = 0.0

    solver = create_solver(sim_cfg, grid)

    cap = bc.before_step_action()
    if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
        potential = potential.astype(state.psi.dtype)
        cap = cap.astype(state.psi.dtype)
    potential = potential + cap

    laser = PulseGaussian(sim_cfg.laser, grid.X, grid.Y)
    spatial_profile = laser._spatial_envelope
    P_zero = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)

    n_steps = int(sim_cfg.solver.total_time / sim_cfg.solver.dt)
    dt = sim_cfg.solver.dt

    print(
        f"  P={P:.4f}  n_steps={n_steps:,}  dt={dt}  "
        f"grid={grid.nx}x{grid.ny}  check_every={scalar_check_every}"
    )

    psi_sq_max_global = 0.0
    t_psi_sq_max_global = 0.0
    scalar_times: list[float] = []
    scalar_psi_sq_max_vals: list[float] = []

    status = "ok"
    diverged_at_step: int | None = None
    diverged_at_t: float | None = None
    last_step = 0

    for step in range(n_steps):
        t = step * dt
        last_step = step

        t_eff = t - sim_cfg.laser.delay
        if t_eff >= 0.0:
            amp = laser._amplitude(t_eff)
            pt = laser._P_time(t_eff)
            temporal = float(amp) * pt
        else:
            temporal = 0.0
        P_total = temporal * spatial_profile if temporal != 0.0 else P_zero

        solver.step(potential, P_total, reservoir, bc, state)

        if step > 0 and step % scalar_check_every == 0:
            psi_sq_max = float(xp.max(xp.abs(state.psi) ** 2))
            t_now = (step + 1) * dt

            if math.isnan(psi_sq_max) or math.isinf(psi_sq_max):
                status = "diverged"
                diverged_at_step = step
                diverged_at_t = t_now
                print(
                    f"  DIVERGED at step={step}, t={t_now:.3f} ps  "
                    f"(P={P:.4f})"
                )
                if save_trace:
                    scalar_times.append(t_now)
                    scalar_psi_sq_max_vals.append(float("nan"))
                if early_stop_on_divergence:
                    break

            if status == "ok":
                if save_trace:
                    scalar_times.append(t_now)
                    scalar_psi_sq_max_vals.append(psi_sq_max)
                if psi_sq_max > psi_sq_max_global:
                    psi_sq_max_global = psi_sq_max
                    t_psi_sq_max_global = t_now

    if status == "ok" and n_steps > 0:
        t_final = n_steps * dt
        already_sampled = scalar_times and scalar_times[-1] >= t_final - 0.5 * dt
        if not already_sampled:
            psi_sq_max_final = float(xp.max(xp.abs(state.psi) ** 2))
            if math.isnan(psi_sq_max_final) or math.isinf(psi_sq_max_final):
                status = "diverged"
                diverged_at_step = n_steps - 1
                diverged_at_t = t_final
                print(
                    f"  DIVERGED (final sample) at t={t_final:.3f} ps  (P={P:.4f})"
                )
                if save_trace:
                    scalar_times.append(t_final)
                    scalar_psi_sq_max_vals.append(float("nan"))
            else:
                if save_trace:
                    scalar_times.append(t_final)
                    scalar_psi_sq_max_vals.append(psi_sq_max_final)
                if psi_sq_max_final > psi_sq_max_global:
                    psi_sq_max_global = psi_sq_max_final
                    t_psi_sq_max_global = t_final

    wall_time = time.monotonic() - wall_start
    print(
        f"  Done  P={P:.4f}  status={status}  "
        f"psi_sq_max={psi_sq_max_global:.6e}  "
        f"wall={wall_time:.1f}s"
    )

    return PowerResult(
        power_index=power_index,
        P=P,
        status=status,
        psi_sq_max=psi_sq_max_global,
        t_psi_sq_max=t_psi_sq_max_global,
        n_steps_total=n_steps,
        last_step_completed=last_step,
        diverged_at_step=diverged_at_step,
        diverged_at_t=diverged_at_t,
        wall_time_seconds=wall_time,
        scalar_times=scalar_times,
        scalar_psi_sq_max=scalar_psi_sq_max_vals,
    )
