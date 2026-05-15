"""Physics simulation kernel for the GPE and reservoir model.

This module only handles the simulation work. It does not know about
Slurm, config-file layout, or pipeline directories.
"""
from __future__ import annotations

import math
import os
import traceback

import numpy as np

from pipeline.config.output_policy import OutputPolicy
from pipeline.simulation.roi import (
    compile_circle_rois,
    compute_roi_scalars,
    scalar_names_for_rois,
)
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.grid.create_grid import create_grid
from polarism.init_condition import make_initial_psi
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.results.storage import create_hdf5_writer
from polarism.results.storage.appendable_hdf5 import compute_batch_size
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver
from tqdm import trange

COND_GROWTH_FACTOR = 1e6
CHECK_EVERY = 50
MIN_CHECK_TIME = 50.0
RNG_SEED = 42

_FIELD_SPECS: dict[str, np.dtype] = {
    "psi": np.dtype(np.complex128),
    "nI": np.dtype(np.float64),
    "nA": np.dtype(np.float64),
    "Pump": np.dtype(np.float64),
}


def check_condensation_kspace(psi: object, xp: object) -> float:
    """Return the k=0 power fraction of *psi* (FFT-based)."""
    psi_k = xp.fft.fft2(psi)
    power = xp.abs(psi_k) ** 2
    k0_power = float(power[0, 0])
    total_power = float(xp.sum(power))
    if total_power < 1e-30:
        return 0.0
    return k0_power / total_power


def _compute_kspace_metrics(psi, k_r, k_nyquist_min: float, xp) -> dict:
    """Compute spectral power distribution metrics for *psi*.

    Parameters
    ----------
    psi : array
        Complex wavefunction.
    k_r : array
        Radial k-values in rad/um for each FFT bin, shape (ny, nx).
    k_nyquist_min : float
        min(π/dx, π/dy) in rad/um.
    xp : module
        Compute backend.

    Returns
    -------
    dict with keys: k0_frac, high_k_frac_0p5_nyq, high_k_frac_0p8_nyq,
    k_peak_um, k_centroid_um.
    """
    psi_k = xp.fft.fft2(psi)
    power = xp.abs(psi_k) ** 2
    total_power = float(xp.sum(power))
    if total_power < 1e-30:
        return {
            "k0_frac": 0.0,
            "high_k_frac_0p5_nyq": 0.0,
            "high_k_frac_0p8_nyq": 0.0,
            "k_peak_um": 0.0,
            "k_centroid_um": 0.0,
        }

    k0_frac = float(power[0, 0]) / total_power
    thresh_0p5 = 0.5 * k_nyquist_min
    thresh_0p8 = 0.8 * k_nyquist_min
    high_k_0p5 = float(xp.sum(power * (k_r > thresh_0p5))) / total_power
    high_k_0p8 = float(xp.sum(power * (k_r > thresh_0p8))) / total_power

    dc_mask = k_r > 0.0
    power_no_dc = power * dc_mask
    total_no_dc = float(xp.sum(power_no_dc))
    if total_no_dc > 1e-30:
        peak_idx = int(xp.argmax(power_no_dc))
        k_peak = float(k_r.ravel()[peak_idx])
        k_centroid = float(xp.sum(k_r * power_no_dc) / total_no_dc)
    else:
        k_peak = 0.0
        k_centroid = 0.0

    return {
        "k0_frac": k0_frac,
        "high_k_frac_0p5_nyq": high_k_0p5,
        "high_k_frac_0p8_nyq": high_k_0p8,
        "k_peak_um": k_peak,
        "k_centroid_um": k_centroid,
    }


def _precompute_spatial_profiles(lasers: list, grid: object) -> list:
    """Precompute the spatial pump profile for each laser."""
    return [laser._P_space(grid.X, grid.Y) for laser in lasers]


def _p_time_pure(
    t: float,
    pulse_separation: float,
    cutoff_sigma: float,
    sigma_time: float,
) -> float:
    """Return the pulse envelope at time t."""
    phase = cutoff_sigma * sigma_time
    n = max(0, round((t - phase) / pulse_separation))
    dt_val = t - n * pulse_separation - phase
    if abs(dt_val) > cutoff_sigma * sigma_time:
        return 0.0
    return math.exp(-0.5 * (dt_val / sigma_time) ** 2)


def _precompute_spatial_maxes(profiles: list, xp: object) -> list:
    """Precompute the peak value of each spatial profile."""
    return [float(xp.max(p)) for p in profiles]


def _compute_pump_fast(
    lasers: list,
    profiles: list,
    spatial_maxes: list,
    t: float,
    xp: object,
    P_zero: object,
) -> tuple:
    """Build the total pump field and per-laser peaks."""
    P_total = P_zero.copy()
    per_laser_max = []
    for i, laser in enumerate(lasers):
        t_eff = t - laser.delay
        if t_eff < 0:
            per_laser_max.append(0.0)
            continue
        amp = laser._amplitude(t_eff)
        pt = _p_time_pure(t_eff, laser.pulse_separation, laser.cutoff_sigma, laser.sigma_time)
        temporal = float(amp) * pt
        if temporal == 0.0:
            per_laser_max.append(0.0)
            continue
        P_total += temporal * profiles[i]
        per_laser_max.append(temporal * spatial_maxes[i])
    return P_total, per_laser_max


def _active_inactive_reservoir_fields(reservoir: object, inactive_zero: object):
    """Return active and inactive reservoir fields for all reservoir models.

    The pipeline stores the active reservoir as ``nA`` in HDF5 for backward
    compatibility, even when the model calls that field ``nR``. Single-reservoir
    models have no inactive field, so a reusable zero array is emitted as ``nI``.
    """
    state = reservoir.get_state()
    if len(state) == 1:
        return state[0], inactive_zero
    if len(state) == 2:
        return state[0], state[1]
    raise ValueError(
        "Reservoir state must contain one active field or active/inactive fields; "
        f"got {len(state)} entries"
    )


def run_simulation_from_config(
    routine_name: str,
    lasers: list,
    cfg: Config,
    output_dir: str,
    output_policy: OutputPolicy | None = None,
    roi_specs: list[dict] | None = None,
) -> tuple[float | None, str, dict]:
    """Run a simulation with an explicit ``Config`` object.

    Parameters
    ----------
    routine_name : str
        Output filename stem.  HDF5 written as
        ``<output_dir>/<routine_name>.h5``, scalar sidecar as
        ``<output_dir>/<routine_name>_scalars.npz``.
    lasers : list
        Laser instances (e.g. ``PulseGaussian``).
    cfg : Config
        Fully constructed simulation config.
    output_dir : str
        Directory for outputs.  Created if it does not already exist.
    output_policy : OutputPolicy or None
        Recording cadence and archival flags.  Defaults are used when ``None``.
    roi_specs : list of dict or None
        Optional circular ROI definitions resolved by the pipeline config layer.

    Returns
    -------
    tuple[float or None, str, dict]
        ``(t_cond, sidecar_path, initial_condition_meta)`` — condensation time
        in ps (or ``None``), the path of the written scalar ``.npz`` sidecar,
        and a dict with initial-condition metadata and spectral metrics.
    """
    if output_policy is None:
        output_policy = OutputPolicy()
    os.makedirs(output_dir, exist_ok=True)

    xp = compute_engine.xp

    grid = create_grid(cfg.grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics)
    potential = create_potential(cfg.potential, grid)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid)

    init_mode = getattr(cfg.physics, "init_mode", "legacy_positive_uniform")
    init_k_cutoff_um = getattr(cfg.physics, "init_k_cutoff_um", None)
    init_seed_cfg = getattr(cfg.physics, "init_seed", None)
    effective_seed = init_seed_cfg if init_seed_cfg is not None else RNG_SEED

    state = SimulationState.__new__(SimulationState)
    state.psi = make_initial_psi(
        xp, grid.ny, grid.nx,
        eps=cfg.physics.init_eps,
        mode=init_mode,
        dx=grid.dx,
        dy=grid.dy,
        k_cutoff_um=init_k_cutoff_um,
        seed=effective_seed,
        cdtype=xp.complex128,
        rdtype=xp.float64,
    )
    state.t = 0.0

    kx_rad = 2.0 * math.pi * xp.fft.fftfreq(grid.nx, d=grid.dx)
    ky_rad = 2.0 * math.pi * xp.fft.fftfreq(grid.ny, d=grid.dy)
    KX_k, KY_k = xp.meshgrid(kx_rad, ky_rad)
    k_r = xp.sqrt(KX_k ** 2 + KY_k ** 2)
    k_nyquist_min = float(min(math.pi / grid.dx, math.pi / grid.dy))

    initial_kspace = _compute_kspace_metrics(state.psi, k_r, k_nyquist_min, xp)
    initial_condition_meta: dict = {
        "init_mode": init_mode,
        "init_eps": cfg.physics.init_eps,
        "init_k_cutoff_um": init_k_cutoff_um,
        "init_seed": init_seed_cfg,
        "initial_k0_frac": initial_kspace["k0_frac"],
        "initial_high_k_frac_0p5_nyq": initial_kspace["high_k_frac_0p5_nyq"],
        "initial_high_k_frac_0p8_nyq": initial_kspace["high_k_frac_0p8_nyq"],
    }

    solver = create_solver(cfg, grid)

    cap = bc.before_step_action()
    if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
        potential = potential.astype(state.psi.dtype)
        cap = cap.astype(state.psi.dtype)
    potential = potential + cap

    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    dt = cfg.solver.dt
    batch_size = compute_batch_size(_FIELD_SPECS, (grid.ny, grid.nx))
    print(
        f"    n_steps={n_steps:,}, dt={dt}, batch={batch_size}, "
        f"field_stride={output_policy.field_record_stride}, "
        f"scalar_stride={output_policy.scalar_record_stride}"
    )

    out_path = os.path.join(output_dir, f"{routine_name}.h5")
    writer = create_hdf5_writer(out_path, batch_size, _FIELD_SPECS, (grid.ny, grid.nx))

    rois = compile_circle_rois(roi_specs or [], grid, xp)
    if rois:
        print(
            "    ROI metrics: "
            + ", ".join(
                f"{roi.id}(r={roi.radius:g}, cells={roi.count})" for roi in rois
            )
        )

    hbar_over_2m = cfg.physics.hbar / (2.0 * cfg.physics.m_eff)
    kinetic_relaxation_eta = getattr(cfg.physics, "kinetic_relaxation_eta", 0.0)

    scalar_names = [
        "psi_sq_max", "nI_max", "nA_max",
        "k0_frac",
        "high_k_frac_0p5_nyq", "high_k_frac_0p8_nyq",
        "k_peak_um", "k_centroid_um",
        "P_max",
        "nyquist_k_um",
        "estimated_nyquist_damping_max",
        "estimated_nyquist_growth_margin_max",
    ]
    scalar_names.extend(scalar_names_for_rois(rois))
    for li in range(len(lasers)):
        scalar_names.append(f"P_max_{li}")
    for name in scalar_names:
        writer.register_scalar(name)

    scalar_acc: dict[str, list[float]] = {name: [] for name in scalar_names}
    scalar_times: list[float] = []

    spatial_profiles = _precompute_spatial_profiles(lasers, grid)
    spatial_maxes = _precompute_spatial_maxes(spatial_profiles, xp)
    P_zero = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)
    inactive_zero = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)

    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))
    t_cond = None
    condensed = False
    last_t = 0.0
    step = 0

    try:
        for step in trange(n_steps, desc=f"  {routine_name}"):
            t = step * dt
            last_t = t

            P_total, per_laser_max = _compute_pump_fast(
                lasers, spatial_profiles, spatial_maxes, t, xp, P_zero
            )
            solver.step(potential, P_total, reservoir, bc, state)

            if step > 0 and step % CHECK_EVERY == 0:
                N_total = float(xp.sum(xp.abs(state.psi) ** 2))
                if math.isnan(N_total) or math.isinf(N_total):
                    raise RuntimeError(
                        f"Numerical divergence at step {step}, t={t:.4f} ps: "
                        f"psi norm is {N_total}"
                    )
                if not condensed and t >= MIN_CHECK_TIME and N_total > N_initial * COND_GROWTH_FACTOR:
                    t_cond = (step + 1) * dt
                    condensed = True
                    print(f"\n    Condensation at t={t:.2f} ps (step {step})")

            record_scalars = step % output_policy.scalar_record_stride == 0
            record_fields = step % output_policy.field_record_stride == 0

            if record_scalars or record_fields:
                nA, nI = _active_inactive_reservoir_fields(reservoir, inactive_zero)
                psi_sq = xp.abs(state.psi) ** 2
                kspace_metrics = _compute_kspace_metrics(state.psi, k_r, k_nyquist_min, xp)
                nA_max_val = float(xp.max(nA))
                damping_nyq = kinetic_relaxation_eta * nA_max_val * hbar_over_2m * k_nyquist_min ** 2
                gain_amp = 0.5 * (cfg.physics.R * nA_max_val - cfg.physics.gamma_C)
                scalars = {
                    "psi_sq_max": float(xp.max(psi_sq)),
                    "nI_max": float(xp.max(nI)),
                    "nA_max": nA_max_val,
                    "k0_frac": kspace_metrics["k0_frac"],
                    "high_k_frac_0p5_nyq": kspace_metrics["high_k_frac_0p5_nyq"],
                    "high_k_frac_0p8_nyq": kspace_metrics["high_k_frac_0p8_nyq"],
                    "k_peak_um": kspace_metrics["k_peak_um"],
                    "k_centroid_um": kspace_metrics["k_centroid_um"],
                    "P_max": float(xp.max(P_total)),
                    "nyquist_k_um": k_nyquist_min,
                    "estimated_nyquist_damping_max": damping_nyq,
                    "estimated_nyquist_growth_margin_max": gain_amp - damping_nyq,
                }
                if rois:
                    scalars.update(
                        compute_roi_scalars(
                            rois, psi_sq, nA, nI, cfg.physics.gamma_C, xp
                        )
                    )
                for li, lp in enumerate(per_laser_max):
                    scalars[f"P_max_{li}"] = lp

            if record_scalars:
                scalar_times.append((step + 1) * dt)
                for name in scalar_names:
                    scalar_acc[name].append(scalars.get(name, 0.0))

            if record_fields:
                nA, nI = _active_inactive_reservoir_fields(reservoir, inactive_zero)
                writer.record(
                    (step + 1) * dt,
                    {"psi": state.psi, "nI": nI, "nA": nA, "Pump": P_total},
                    scalars,
                )

    except Exception as e:
        print(f"\n    ERROR at step {step}, t={last_t:.2f} ps: {e}")
        traceback.print_exc()
        raise
    finally:
        print("    Closing HDF5 writer ...")
        writer.close()
        print("    HDF5 finalized.")

    sidecar_path = os.path.join(output_dir, f"{routine_name}_scalars.npz")
    if not os.path.isdir(output_dir):
        raise RuntimeError(
            "Simulation output directory disappeared before scalar sidecar write: "
            f"{output_dir}"
        )
    if not os.path.isfile(out_path):
        raise RuntimeError(
            "HDF5 output disappeared before scalar sidecar write: "
            f"{out_path}"
        )
    np.savez_compressed(
        sidecar_path,
        time=np.array(scalar_times, dtype=np.float64),
        **{k: np.array(v, dtype=np.float64) for k, v in scalar_acc.items()},
    )
    print(f"    Scalar sidecar: {sidecar_path}")

    print(
        f"    -> {out_path}  ({writer.total} frames, "
        f"last_t={last_t:.1f} ps, t_cond={t_cond})"
    )
    return t_cond, sidecar_path, initial_condition_meta
