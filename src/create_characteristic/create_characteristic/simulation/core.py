"""Pure simulation kernel for a single 2D characteristic map point.

This module contains only physics logic — no file I/O, no Slurm, no
config-file paths.  The caller supplies a fully parsed config dict and
the point parameters; this module returns a structured result.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from tqdm import trange

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

from create_characteristic.config.builder import build_point_config

RNG_SEED = 42


@dataclass
class PointResult:
    """Scalar results for one 2D characteristic map point."""

    point_index: int
    energy_index: int
    sep_index: int
    pulse_energy: float
    pulse_separation: float
    total_time: float
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


def run_grid_point(
    point_index: int,
    energy_index: int,
    sep_index: int,
    pulse_energy: float,
    pulse_separation: float,
    total_time: float,
    cfg: dict[str, Any],
    scalar_check_every: int = 100,
    early_stop_on_divergence: bool = True,
    save_trace: bool = False,
) -> PointResult:
    """Simulate polariton dynamics at one 2D grid point and return scalar results.

    Parameters
    ----------
    point_index : int
        Flat index of this point in the grid.
    energy_index : int
        Index along the pulse energy axis.
    sep_index : int
        Index along the pulse separation axis.
    pulse_energy : float
        Pulse energy for this point.
    pulse_separation : float
        Pulse separation in ps for this point.
    total_time : float
        Simulation duration in ps.
    cfg : dict
        Full parsed config dict.
    scalar_check_every : int
        Record psi_sq_max every this many simulation steps.
    early_stop_on_divergence : bool
        Terminate the loop immediately on NaN/Inf detection.
    save_trace : bool
        Accumulate the full per-step scalar trace in the result.

    Returns
    -------
    PointResult
        Scalar results; ``status`` is ``'ok'`` or ``'diverged'``.
    """
    wall_start = time.monotonic()

    sim_cfg = build_point_config(cfg, pulse_energy, pulse_separation, total_time)
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
        f"  E={pulse_energy:.1f}  sep={pulse_separation:.1f}ps  "
        f"n_steps={n_steps:,}  dt={dt}  grid={grid.nx}x{grid.ny}"
    )

    psi_sq_max_global = 0.0
    t_psi_sq_max_global = 0.0
    scalar_times: list[float] = []
    scalar_psi_sq_max_vals: list[float] = []

    status = "ok"
    diverged_at_step: int | None = None
    diverged_at_t: float | None = None
    last_step = 0

    pbar_desc = f"  E={pulse_energy:.0f} sep={pulse_separation:.0f}"
    with trange(n_steps, desc=pbar_desc, dynamic_ncols=True) as pbar:
        for step in pbar:
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
                    pbar.write(
                        f"  DIVERGED at step={step}, t={t_now:.3f} ps  "
                        f"(E={pulse_energy:.1f}, sep={pulse_separation:.1f})"
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
                    pbar.set_postfix(
                        psi_sq=f"{psi_sq_max:.2e}",
                        max=f"{psi_sq_max_global:.2e}",
                    )

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
                    f"  DIVERGED (final sample) at t={t_final:.3f} ps  "
                    f"(E={pulse_energy:.1f}, sep={pulse_separation:.1f})"
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
        f"  Done  E={pulse_energy:.1f}  sep={pulse_separation:.1f}  "
        f"status={status}  psi_sq_max={psi_sq_max_global:.6e}  wall={wall_time:.1f}s"
    )

    return PointResult(
        point_index=point_index,
        energy_index=energy_index,
        sep_index=sep_index,
        pulse_energy=pulse_energy,
        pulse_separation=pulse_separation,
        total_time=total_time,
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
