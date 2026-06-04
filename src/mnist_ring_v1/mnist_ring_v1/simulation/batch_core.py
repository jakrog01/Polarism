"""Core simulation loop for one image: reset state, run, extract features."""
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

from mnist_ring_v1.simulation.readout import extract_kspace_feature

_COND_PSI_SQ_THRESHOLD = 5e-2
_COND_GROWTH_FACTOR = 1e6
_CHECK_EVERY = 50
# Global detector minimum time: kept at 50 ps for backward compat.
# Central detector starts from 0 ps to catch any pre-trigger central condensation.
_MIN_CHECK_TIME_GLOBAL = 50.0
_MIN_CHECK_TIME_CENTRAL = 0.0


def _kspace_metrics(psi: Any, k_r: Any, k_nyq: float, xp: Any) -> dict[str, float]:
    psi_k = xp.fft.fft2(psi)
    power = xp.abs(psi_k) ** 2
    total = float(xp.sum(power))
    if total < 1e-30:
        return {"k0_frac": 0.0, "high_k_frac_0p8_nyq": 0.0, "k_peak_um": 0.0, "k_centroid_um": 0.0}

    k0_frac = float(power[0, 0]) / total
    high_k = float(xp.sum(power * (k_r > 0.8 * k_nyq))) / total

    dc_mask = k_r > 0.0
    p_no_dc = power * dc_mask
    total_no_dc = float(xp.sum(p_no_dc))
    if total_no_dc > 1e-30:
        k_peak = float(k_r.ravel()[int(xp.argmax(p_no_dc))])
        k_cent = float(xp.sum(k_r * p_no_dc) / total_no_dc)
    else:
        k_peak = k_cent = 0.0

    return {
        "k0_frac": k0_frac,
        "high_k_frac_0p8_nyq": high_k,
        "k_peak_um": k_peak,
        "k_centroid_um": k_cent,
    }


class SharedSimResources:
    """Reusable GPU resources (grid, BC, potential, solver) shared across images."""

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
        self.k_nyq = float(min(math.pi / self.grid.dx, math.pi / self.grid.dy))

        self.solver = create_solver(cfg, self.grid)


def _build_central_mask(grid: Any, central_roi_um: float, xp: Any) -> Any:
    """Boolean mask True inside a circle of radius `central_roi_um` at origin."""
    r2 = grid.X ** 2 + grid.Y ** 2
    return r2 <= central_roi_um ** 2


def simulate_one_image(
    resources: SharedSimResources,
    lasers: list[Any],
    readout_window_start_ps: float,
    readout_window_end_ps: float,
    kspace_crop_size: int,
    kspace_normalization: str,
    readout_stride_steps: int = 100,
    trigger_delay_ps: float | None = None,
    central_roi_um: float = 6.0,
    stop_time_ps: float | None = None,
) -> dict[str, Any]:
    """Run one image simulation and extract features.

    Resets psi and reservoir each call; reuses grid, BC, potential, solver.

    Parameters
    ----------
    resources
        Shared GPU resources from SharedSimResources.
    lasers
        17 laser instances for this image.
    readout_window_start_ps, readout_window_end_ps
        Time window for k-space readout (ps from start of sim).
    kspace_crop_size
        Pixels per side of cropped far-field.
    kspace_normalization
        Passed to extract_kspace_feature.
    readout_stride_steps
        Record one far-field frame every N solver steps inside the readout window.
    trigger_delay_ps
        Delay of the central trigger in ps.  Used to flag pre-trigger central
        condensation.  None disables the pre-trigger flag.
    central_roi_um
        Radius of the central detection region in μm.  Should be smaller than
        the ring radius (20 μm) to avoid counting ring-spot condensation.
        Default 6 μm.
    stop_time_ps
        Stop simulation after this time (ps).  None means run to total_time.
        Useful for calibration guard passes that only need to check up to T_max.

    Returns
    -------
    dict with keys:
        Backward-compatible keys (global detector):
            'kspace_feature', 'psi_sq_max', 'k0_frac', 'k_peak_um',
            'k_centroid_um', 'high_k_frac_0p8_nyq', 't_cond', 'condensed',
            'readout_frames'
        New central-detector keys:
            'psi_sq_central_max'         : float  max |psi|^2 in central ROI
            'central_condensed'          : bool   central ROI exceeded threshold
            't_cond_central'             : float | None  time of first central cond
            'pretrigger_central_condensed': bool  central_condensed before trigger
        Detector metadata:
            'detector_config'            : dict
    """
    cfg = resources.cfg
    xp = compute_engine.xp

    reservoir = create_reservoir(cfg.reservoir, cfg.physics, resources.grid)

    state = SimulationState.__new__(SimulationState)
    state.psi = make_initial_psi(
        xp, resources.grid.ny, resources.grid.nx,
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

    total_time = cfg.solver.total_time if stop_time_ps is None else min(float(stop_time_ps), cfg.solver.total_time)
    n_steps = int(total_time / cfg.solver.dt)
    dt = cfg.solver.dt
    readout_stride_steps = max(1, int(readout_stride_steps))

    spatial_profiles = [laser._spatial_envelope for laser in lasers]
    P_zero = xp.zeros((resources.grid.ny, resources.grid.nx), dtype=xp.float64)

    central_mask = _build_central_mask(resources.grid, central_roi_um, xp)

    t_cond_global: float | None = None
    global_condensed = False
    t_cond_central: float | None = None
    central_condensed = False
    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))

    kspace_accum: list[np.ndarray] = []
    psi_sq_global_max = 0.0
    psi_sq_central_max = 0.0
    readout_frames = 0
    last_kspace_metrics: dict[str, float] = {}

    for step in range(n_steps):
        t = step * dt

        P_total = P_zero.copy()
        for i, laser in enumerate(lasers):
            t_eff = t - laser.delay
            if t_eff < 0:
                continue
            amp = laser._amplitude(t_eff)
            pt = laser._P_time(t_eff)
            temporal = float(amp) * pt
            if temporal != 0.0:
                P_total = P_total + temporal * spatial_profiles[i]

        resources.solver.step(resources.potential, P_total, reservoir, resources.bc, state)

        if step > 0 and step % _CHECK_EVERY == 0:
            psi_sq = xp.abs(state.psi) ** 2
            N_total = float(xp.sum(psi_sq))

            psi_sq_max_global = float(xp.max(psi_sq))
            psi_sq_global_max = max(psi_sq_global_max, psi_sq_max_global)

            psi_sq_central_step = float(xp.max(psi_sq * central_mask))
            psi_sq_central_max = max(psi_sq_central_max, psi_sq_central_step)

            t_next = (step + 1) * dt

            if not global_condensed and t >= _MIN_CHECK_TIME_GLOBAL:
                if psi_sq_max_global > _COND_PSI_SQ_THRESHOLD or N_total > N_initial * _COND_GROWTH_FACTOR:
                    t_cond_global = t_next
                    global_condensed = True

            if not central_condensed and t >= _MIN_CHECK_TIME_CENTRAL:
                if psi_sq_central_step > _COND_PSI_SQ_THRESHOLD:
                    t_cond_central = t_next
                    central_condensed = True

        t_next = (step + 1) * dt
        if (
            readout_window_start_ps <= t_next <= readout_window_end_ps
            and step % readout_stride_steps == 0
        ):
            feat = extract_kspace_feature(
                state.psi, xp, crop_size=kspace_crop_size,
                normalization=kspace_normalization,
            )
            kspace_accum.append(feat)
            readout_frames += 1
            last_kspace_metrics = _kspace_metrics(state.psi, resources.k_r, resources.k_nyq, xp)

    if kspace_accum:
        kspace_feature = np.mean(kspace_accum, axis=0)
    else:
        kspace_feature = np.zeros(kspace_crop_size * kspace_crop_size, dtype=np.float64)

    pretrigger_central = (
        central_condensed
        and trigger_delay_ps is not None
        and t_cond_central is not None
        and t_cond_central < trigger_delay_ps
    )

    return {
        "kspace_feature": kspace_feature,
        "psi_sq_max": psi_sq_global_max,
        "k0_frac": last_kspace_metrics.get("k0_frac", 0.0),
        "k_peak_um": last_kspace_metrics.get("k_peak_um", 0.0),
        "k_centroid_um": last_kspace_metrics.get("k_centroid_um", 0.0),
        "high_k_frac_0p8_nyq": last_kspace_metrics.get("high_k_frac_0p8_nyq", 0.0),
        "t_cond": t_cond_global,
        "condensed": global_condensed,
        "readout_frames": readout_frames,
        "psi_sq_central_max": psi_sq_central_max,
        "central_condensed": central_condensed,
        "t_cond_central": t_cond_central,
        "pretrigger_central_condensed": pretrigger_central,
        "detector_config": {
            "central_roi_um": central_roi_um,
            "cond_threshold": _COND_PSI_SQ_THRESHOLD,
            "min_check_time_global_ps": _MIN_CHECK_TIME_GLOBAL,
            "min_check_time_central_ps": _MIN_CHECK_TIME_CENTRAL,
            "trigger_delay_ps": trigger_delay_ps,
        },
    }
