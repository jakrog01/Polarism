"""Spatial snapshot helpers: run simulation and record state at fixed time points."""
from __future__ import annotations

from typing import Any

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.init_condition import make_initial_psi
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState

from mnist_ring_v1.simulation.batch_core import SharedSimResources

_COND_PSI_SQ_THRESHOLD = 5e-2
_COND_GROWTH_FACTOR = 1e6
_CHECK_EVERY = 50


def _to_np(arr: Any) -> np.ndarray:
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _downsample_2d_mean(arr: Any, factor: int, xp: Any) -> np.ndarray:
    ny, nx = arr.shape
    ny2, nx2 = ny // factor, nx // factor
    crop = xp.ascontiguousarray(arr[: ny2 * factor, : nx2 * factor])
    return _to_np(xp.mean(xp.reshape(crop, (ny2, factor, nx2, factor)), axis=(1, 3)))


def _angular_profile(
    field_np: np.ndarray,
    x_np: np.ndarray,
    y_np: np.ndarray,
    ring_r_um: float,
    dr_um: float,
    n_bins: int,
) -> np.ndarray:
    """Angular-bin average of field at ring radius ring_r_um ± dr_um."""
    r = np.sqrt(x_np ** 2 + y_np ** 2)
    mask = (r > ring_r_um - dr_um) & (r <= ring_r_um + dr_um)
    theta = np.arctan2(y_np[mask], x_np[mask])
    vals = field_np[mask]
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    profile, _ = np.histogram(theta, bins=edges, weights=vals)
    counts, _ = np.histogram(theta, bins=edges)
    with np.errstate(invalid="ignore"):
        profile = np.where(counts > 0, profile / counts, 0.0)
    return profile


def simulate_one_image_with_snapshots(
    resources: SharedSimResources,
    lasers: list[Any],
    snapshot_times_ps: np.ndarray,
    ring_radius_um: float,
    downsample_factor: int = 8,
    n_angular_bins: int = 64,
    angular_dr_um: float = 2.0,
) -> dict[str, Any]:
    """Run image simulation and record spatial state at requested time points.

    Parameters
    ----------
    resources
        Shared GPU resources (grid, BC, solver).
    lasers
        Laser list for this image (17 ring+trigger or custom).
    snapshot_times_ps
        Target times in ps.  Each snapshot is captured at the first solver step
        where ``t_next >= target``.  Must be within solver total_time.
    ring_radius_um
        Ring radius for the n_R angular profile extraction.
    downsample_factor
        Spatial coarsegraining factor.  Default 8 → 1024/8 = 128×128 output.
    n_angular_bins
        Angular bins for the ring profile.
    angular_dr_um
        Radial half-width of the annular band used for the profile.

    Returns
    -------
    dict with keys:
        times_ps                 : recorded times (ps)
        psi_sq_downsampled       : (n_snaps, ny_ds, nx_ds)
        nR_downsampled           : (n_snaps, ny_ds, nx_ds)
        nI_downsampled           : (n_snaps, ny_ds, nx_ds)
        pump_downsampled         : (n_snaps, ny_ds, nx_ds)
        ring_nR_angular_profile  : (n_snaps, n_angular_bins)
        condensed                : bool
        t_cond                   : float | None
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

    snap_sorted = np.sort(np.asarray(snapshot_times_ps, dtype=np.float64))
    stop_time = float(snap_sorted[-1]) if len(snap_sorted) > 0 else cfg.solver.total_time
    stop_time = min(stop_time, cfg.solver.total_time)
    n_steps = int(round(stop_time / cfg.solver.dt))
    dt = cfg.solver.dt

    spatial_profiles = [laser._spatial_envelope for laser in lasers]
    P_zero = xp.zeros((resources.grid.ny, resources.grid.nx), dtype=xp.float64)

    x_np = _to_np(resources.grid.X)
    y_np = _to_np(resources.grid.Y)

    snap_idx = 0
    n_snaps = len(snap_sorted)
    recorded_times: list[float] = []
    psi_sq_ds_list: list[np.ndarray] = []
    nR_ds_list: list[np.ndarray] = []
    nI_ds_list: list[np.ndarray] = []
    pump_ds_list: list[np.ndarray] = []
    angular_list: list[np.ndarray] = []

    t_cond: float | None = None
    condensed = False
    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))

    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt

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
            N_now = float(xp.sum(psi_sq))
            psi_sq_max = float(xp.max(psi_sq))
            if not condensed:
                if psi_sq_max > _COND_PSI_SQ_THRESHOLD or N_now > N_initial * _COND_GROWTH_FACTOR:
                    t_cond = t_next
                    condensed = True

        while snap_idx < n_snaps and t_next >= snap_sorted[snap_idx]:
            psi_sq_now = xp.abs(state.psi) ** 2
            nR = getattr(reservoir, "nR", None)
            nI = getattr(reservoir, "nI", None)

            psi_sq_ds_list.append(_downsample_2d_mean(psi_sq_now, downsample_factor, xp))
            nR_ds_list.append(
                _downsample_2d_mean(nR, downsample_factor, xp)
                if nR is not None
                else np.zeros_like(psi_sq_ds_list[-1])
            )
            nI_ds_list.append(
                _downsample_2d_mean(nI, downsample_factor, xp)
                if nI is not None
                else np.zeros_like(psi_sq_ds_list[-1])
            )
            pump_ds_list.append(_downsample_2d_mean(P_total, downsample_factor, xp))

            nR_np = _to_np(nR) if nR is not None else np.zeros_like(x_np)
            angular_list.append(
                _angular_profile(nR_np, x_np, y_np, ring_radius_um, angular_dr_um, n_angular_bins)
            )
            recorded_times.append(float(t_next))
            snap_idx += 1

    def _stack(lst: list[np.ndarray]) -> np.ndarray:
        return np.stack(lst, axis=0) if lst else np.zeros((0,))

    return {
        "times_ps": np.array(recorded_times),
        "psi_sq_downsampled": _stack(psi_sq_ds_list),
        "nR_downsampled": _stack(nR_ds_list),
        "nI_downsampled": _stack(nI_ds_list),
        "pump_downsampled": _stack(pump_ds_list),
        "ring_nR_angular_profile": _stack(angular_list),
        "condensed": condensed,
        "t_cond": t_cond,
    }
