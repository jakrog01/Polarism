"""WTA core simulation: shared GPU resources and single-image runner."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.grid.create_grid import create_grid
from polarism.init_condition import make_initial_psi
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver


_COND_PSI_SQ_THRESHOLD = 5e-2
_COND_GROWTH_FACTOR = 1e6
_CHECK_EVERY = 50


class SharedSimResources:
    """Reusable GPU resources (grid, BC, potential, solver)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        xp = compute_engine.xp

        self.grid = create_grid(cfg.grid)
        bc = BoundaryCondition(self.grid, cfg.boundary_condition, cfg.physics)
        potential = create_potential(cfg.potential, self.grid)
        cap = bc.before_step_action()
        if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
            potential = potential.astype(xp.complex128)
            cap = cap.astype(xp.complex128)
        self.potential = potential + cap
        self.bc = bc

        kx = 2.0 * math.pi * xp.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        ky = 2.0 * math.pi * xp.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        KX, KY = xp.meshgrid(kx, ky)
        self.k_r = xp.sqrt(KX ** 2 + KY ** 2)

        self.solver = create_solver(cfg, self.grid)


def simulate_one_image_wta(
    resources: SharedSimResources,
    lasers: list[Any],
    class_masks: list[Any],
    threshold_cond: float,
    readout_window_start_ps: float,
    readout_window_end_ps: float,
    readout_stride_steps: int = 100,
    power_per_class: list[float] | None = None,
) -> dict[str, Any]:
    """Run WTA simulation for one image and return readout.

    Parameters
    ----------
    resources
        Shared GPU resources.
        lasers
            n_classes single-pulse class lasers with class-specific pulse energy.
    class_masks
        Precomputed ROI masks per class.
    threshold_cond
        Minimum ROI integral to count as condensed (from calibration).
    readout_window_start_ps, readout_window_end_ps
        Time window for ROI integral accumulation.
    readout_stride_steps
        Sample ROI integrals every N steps inside the window.
    power_per_class
        Per-class pump powers (for metadata logging).

    Returns
    -------
    dict with keys:
        class_roi_integrals  : list[float]
        winner               : int
        margin               : float
        no_cond              : bool
        multi_cond           : bool
        power_per_class      : list[float]
        t_cond               : float | None
        condensed            : bool
        readout_frames       : int
    """
    cfg = resources.cfg
    xp = compute_engine.xp

    reservoir = create_reservoir(cfg.reservoir, cfg.physics, resources.grid)

    state = SimulationState.__new__(SimulationState)
    state.psi = make_initial_psi(
        xp,
        resources.grid.ny,
        resources.grid.nx,
        eps=cfg.physics.init_eps,
        mode=getattr(cfg.physics, "init_mode", "filtered_complex_gaussian"),
        dx=resources.grid.dx,
        dy=resources.grid.dy,
        k_cutoff_um=getattr(cfg.physics, "init_k_cutoff_um", None),
        seed=int(getattr(cfg.physics, "init_seed", 42)),
        cdtype=xp.complex128,
        rdtype=xp.float64,
    )
    state.t = 0.0

    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    dt = cfg.solver.dt
    readout_stride_steps = max(1, int(readout_stride_steps))

    spatial_profiles = [laser._spatial_envelope for laser in lasers]
    P_zero = xp.zeros((resources.grid.ny, resources.grid.nx), dtype=xp.float64)

    t_cond: float | None = None
    condensed = False
    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))

    roi_accum: list[list[float]] = []
    readout_frames = 0

    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt

        P_total = P_zero.copy()
        for i, laser in enumerate(lasers):
            t_eff = t - laser.delay
            if t_eff < 0.0:
                continue
            temporal = float(laser._amplitude(t_eff)) * float(laser._P_time(t_eff))
            if temporal != 0.0:
                P_total = P_total + temporal * spatial_profiles[i]

        resources.solver.step(resources.potential, P_total, reservoir, resources.bc, state)

        if step > 0 and step % _CHECK_EVERY == 0:
            psi_sq = xp.abs(state.psi) ** 2
            N_now = float(xp.sum(psi_sq))
            psi_sq_max = float(xp.max(psi_sq))
            if not condensed:
                if psi_sq_max > _COND_PSI_SQ_THRESHOLD or N_now > N_initial * _COND_GROWTH_FACTOR:
                    t_cond = t_next
                    condensed = True

        if (
            readout_window_start_ps <= t_next <= readout_window_end_ps
            and step % readout_stride_steps == 0
        ):
            psi_sq_now = xp.abs(state.psi) ** 2
            frame_integrals = [float(xp.sum(psi_sq_now * mask)) for mask in class_masks]
            roi_accum.append(frame_integrals)
            readout_frames += 1

    if roi_accum:
        mean_integrals = list(np.mean(roi_accum, axis=0))
    else:
        mean_integrals = [0.0] * len(class_masks)

    integrals_arr = np.array(mean_integrals)
    winner = int(np.argmax(integrals_arr))
    sorted_desc = np.sort(integrals_arr)[::-1]
    i_winner = float(sorted_desc[0])
    i_second = float(sorted_desc[1]) if len(sorted_desc) > 1 else 0.0
    margin = i_winner / max(i_second, 1e-30)
    no_cond = i_winner < threshold_cond
    multi_cond = (i_second / max(i_winner, 1e-30)) > 0.5

    return {
        "class_roi_integrals": mean_integrals,
        "winner": winner,
        "margin": float(margin),
        "no_cond": bool(no_cond),
        "multi_cond": bool(multi_cond),
        "power_per_class": power_per_class or [],
        "t_cond": t_cond,
        "condensed": condensed,
        "readout_frames": readout_frames,
    }
