"""Shared simulation core for scenario-based MNIST spacetime experiments."""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np

from mnist_common.io.atomic import atomic_write_json, atomic_write_npz
from mnist_common.simulation.roi import circular_roi_mask
from mnist_spacetime_v1.simulation.lasers import build_lasers
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    ComputeEngineParameters,
    Config,
    GridParameters,
    PhysicsConstants,
    PotentialParameters,
    ReservoirParameters,
    ResultParameters,
    SolverParameters,
)
from polarism.grid.create_grid import create_grid
from polarism.init_condition import make_initial_psi
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver


_COND_PSI_SQ_THRESHOLD = 5e-2
_COND_GROWTH_FACTOR = 1e6
_CHECK_EVERY = 50


def build_polarism_config(cfg: dict[str, Any], use_gpu: bool = True) -> Config:
    """Build a Polarism Config from mnist_spacetime_v1 YAML."""
    g = cfg["global"]["grid"]
    p = cfg["global"]["physics"]
    s = cfg["global"]["solver"]
    bc = cfg["global"].get("boundary_condition", {})
    r = cfg["global"].get("reservoir", {})

    return Config(
        grid=GridParameters(
            nx=int(g["nx"]),
            ny=int(g["ny"]),
            lx=float(g["lx"]),
            ly=float(g["ly"]),
            grid_type=str(g.get("grid_type", "periodic")),
        ),
        boundary_condition=BoundaryConditionParameters(
            profile_type=str(bc.get("profile_type", "sin2")),
            strength=float(bc.get("strength", 5.0)),
            absorption=str(bc.get("absorption", "cap")),
            mask_width_percent=float(bc.get("mask_width_percent", 0.18)),
        ),
        potential=PotentialParameters(potential_type="zero"),
        physics=PhysicsConstants(
            hbar=float(p["hbar"]),
            m_eff=float(p["m_eff"]),
            gamma_C=float(p["gamma_C"]),
            gamma_R=float(p["gamma_R"]),
            gamma_I=float(p.get("gamma_I", 0.001)),
            g_C=float(p["g_C"]),
            g_R=float(p["g_R"]),
            g_I=float(p.get("g_I", 0.0)),
            R=float(p["R"]),
            kappa=float(p.get("kappa", 0.05)),
            kinetic_relaxation_eta=float(p.get("kinetic_relaxation_eta", 1e-5)),
            reservoir_diffusion_I=float(p.get("reservoir_diffusion_I", 0.0)),
            reservoir_diffusion_R=float(p.get("reservoir_diffusion_R", 0.0)),
            init_mode=str(p.get("init_mode", "filtered_complex_gaussian")),
            init_k_cutoff_um=float(p.get("init_k_cutoff_um", 3.0)),
            init_seed=int(p.get("init_seed", 42)),
            init_eps=float(p.get("init_eps", 1e-3)),
        ),
        reservoir=ReservoirParameters(
            reservoir_type=str(r.get("reservoir_type", "quadratic-double")),
        ),
        solver=SolverParameters(
            total_time=float(s["total_time"]),
            dt=float(s["dt"]),
            method=str(s.get("method", "rk4-cuda")),
            laplacian=str(s.get("laplacian", "isotropic-9pt")),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=use_gpu),
    )


class SharedScenarioResources:
    """Reusable GPU resources for a sequence of mechanism scenarios."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        xp = compute_engine.xp

        self.grid = create_grid(cfg.grid)
        self.bc = BoundaryCondition(self.grid, cfg.boundary_condition, cfg.physics)
        potential = create_potential(cfg.potential, self.grid)
        cap = self.bc.before_step_action()
        if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
            potential = potential.astype(xp.complex128)
            cap = cap.astype(xp.complex128)
        self.potential = potential + cap

        kx = 2.0 * math.pi * xp.fft.fftfreq(self.grid.nx, d=self.grid.dx)
        ky = 2.0 * math.pi * xp.fft.fftfreq(self.grid.ny, d=self.grid.dy)
        KX, KY = xp.meshgrid(kx, ky)
        self.k_r = xp.sqrt(KX ** 2 + KY ** 2)

        self.solver = create_solver(cfg, self.grid)


def _initial_state(resources: SharedScenarioResources) -> SimulationState:
    cfg = resources.cfg
    xp = compute_engine.xp
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
    return state


def _roi_masks(resources: SharedScenarioResources, roi_defs: list[dict[str, Any]]) -> list[Any]:
    xp = compute_engine.xp
    masks = []
    for roi in roi_defs:
        if roi.get("shape", "circle") != "circle":
            raise ValueError(f"Unsupported ROI shape for {roi.get('id')}: {roi.get('shape')}")
        masks.append(
            circular_roi_mask(
                resources.grid.X,
                resources.grid.Y,
                float(roi["x0"]),
                float(roi["y0"]),
                float(roi["radius"]),
                xp,
            )
        )
    return masks


def _laser_pump_at(lasers: list[Any], spatial_profiles: list[Any], zero: Any, t: float) -> tuple[Any, list[float]]:
    xp = compute_engine.xp
    p_total = zero.copy()
    p_max_by_laser: list[float] = []
    for laser, spatial in zip(lasers, spatial_profiles):
        t_eff = t - laser.delay
        if t_eff < 0.0:
            p_max_by_laser.append(0.0)
            continue
        temporal = float(laser._amplitude(t_eff)) * float(laser._P_time(t_eff))
        if temporal != 0.0:
            contrib = temporal * spatial
            p_total = p_total + contrib
            p_max_by_laser.append(float(xp.max(contrib)))
        else:
            p_max_by_laser.append(0.0)
    return p_total, p_max_by_laser


def run_scenario(
    resources: SharedScenarioResources,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    run_dir: str,
) -> dict[str, Any]:
    """Run one expanded scenario and persist ROI trace outputs."""
    xp = compute_engine.xp
    sim_cfg = resources.cfg
    output_cfg = cfg.get("output", {})
    stride = max(1, int(output_cfg.get("scalar_stride_steps", 100)))
    save_final_fields = bool(output_cfg.get("save_final_downsampled_fields", True))
    downsample = max(1, int(output_cfg.get("field_downsample", 8)))

    lasers = build_lasers(scenario["lasers"], resources.grid.X, resources.grid.Y)
    spatial_profiles = [laser._spatial_envelope for laser in lasers]
    zero = xp.zeros((resources.grid.ny, resources.grid.nx), dtype=xp.float64)
    masks = _roi_masks(resources, scenario["rois"])

    reservoir = create_reservoir(sim_cfg.reservoir, sim_cfg.physics, resources.grid)
    state = _initial_state(resources)

    n_steps = int(sim_cfg.solver.total_time / sim_cfg.solver.dt)
    dt = float(sim_cfg.solver.dt)
    dxdy = float(resources.grid.dx * resources.grid.dy)
    n_initial = float(xp.sum(xp.abs(state.psi) ** 2))

    t_cond: float | None = None
    condensed = False
    dose = 0.0

    times: list[float] = []
    psi_sq_max: list[float] = []
    nR_max: list[float] = []
    nI_max: list[float] = []
    pump_max: list[float] = []
    pump_area_integral: list[float] = []
    cumulative_dose: list[float] = []
    roi_psi_integrals: list[list[float]] = []
    roi_psi_means: list[list[float]] = []
    roi_nR_means: list[list[float]] = []
    roi_nI_means: list[list[float]] = []
    roi_emission_integrals: list[list[float]] = []
    laser_pmax: list[list[float]] = []

    t_start = time.monotonic()
    for step in range(n_steps):
        t = step * dt
        t_next = (step + 1) * dt
        p_total, p_by_laser = _laser_pump_at(lasers, spatial_profiles, zero, t)
        dose += float(xp.sum(p_total)) * dxdy * dt

        resources.solver.step(resources.potential, p_total, reservoir, resources.bc, state)

        should_sample = (step % stride == 0) or (step == n_steps - 1)
        if step > 0 and step % _CHECK_EVERY == 0:
            psi_sq = xp.abs(state.psi) ** 2
            n_now = float(xp.sum(psi_sq))
            psi_max_now = float(xp.max(psi_sq))
            if not condensed and (
                psi_max_now > _COND_PSI_SQ_THRESHOLD
                or n_now > n_initial * _COND_GROWTH_FACTOR
            ):
                condensed = True
                t_cond = t_next

        if should_sample:
            psi_sq_now = xp.abs(state.psi) ** 2
            nR, nI = reservoir.get_reservoir_densities()
            times.append(t_next)
            psi_sq_max.append(float(xp.max(psi_sq_now)))
            nR_max.append(float(xp.max(nR)))
            nI_max.append(float(xp.max(nI)))
            pump_max.append(float(xp.max(p_total)))
            pump_area_integral.append(float(xp.sum(p_total)) * dxdy)
            cumulative_dose.append(dose)
            laser_pmax.append(p_by_laser)

            frame_psi_int = []
            frame_psi_mean = []
            frame_nR_mean = []
            frame_nI_mean = []
            frame_emission = []
            for mask in masks:
                n_mask = max(float(xp.sum(mask)), 1.0)
                psi_int = float(xp.sum(psi_sq_now * mask)) * dxdy
                frame_psi_int.append(psi_int)
                frame_psi_mean.append(float(xp.sum(psi_sq_now * mask)) / n_mask)
                frame_nR_mean.append(float(xp.sum(nR * mask)) / n_mask)
                frame_nI_mean.append(float(xp.sum(nI * mask)) / n_mask)
                frame_emission.append(float(xp.sum(sim_cfg.physics.R * nR * psi_sq_now * mask)) * dxdy)
            roi_psi_integrals.append(frame_psi_int)
            roi_psi_means.append(frame_psi_mean)
            roi_nR_means.append(frame_nR_mean)
            roi_nI_means.append(frame_nI_mean)
            roi_emission_integrals.append(frame_emission)

    elapsed = time.monotonic() - t_start

    trace_path = os.path.join(run_dir, "traces", f"{scenario['name']}.npz")
    arrays: dict[str, np.ndarray] = {
        "time_ps": np.asarray(times, dtype=np.float64),
        "psi_sq_max": np.asarray(psi_sq_max, dtype=np.float64),
        "nR_max": np.asarray(nR_max, dtype=np.float64),
        "nI_max": np.asarray(nI_max, dtype=np.float64),
        "pump_max": np.asarray(pump_max, dtype=np.float64),
        "pump_area_integral": np.asarray(pump_area_integral, dtype=np.float64),
        "cumulative_dose": np.asarray(cumulative_dose, dtype=np.float64),
        "roi_psi_integrals": np.asarray(roi_psi_integrals, dtype=np.float64),
        "roi_psi_means": np.asarray(roi_psi_means, dtype=np.float64),
        "roi_nR_means": np.asarray(roi_nR_means, dtype=np.float64),
        "roi_nI_means": np.asarray(roi_nI_means, dtype=np.float64),
        "roi_emission_integrals": np.asarray(roi_emission_integrals, dtype=np.float64),
        "laser_pmax": np.asarray(laser_pmax, dtype=np.float64),
        "roi_ids": np.asarray([r["id"] for r in scenario["rois"]]),
        "laser_ids": np.asarray([l["id"] for l in scenario["lasers"]]),
    }

    if save_final_fields:
        sl = (slice(None, None, downsample), slice(None, None, downsample))
        nR_final, nI_final = reservoir.get_reservoir_densities()
        arrays["final_psi_sq_ds"] = compute_engine.to_cpu((xp.abs(state.psi) ** 2)[sl])
        arrays["final_nR_ds"] = compute_engine.to_cpu(nR_final[sl])
        arrays["final_nI_ds"] = compute_engine.to_cpu(nI_final[sl])

    arrays = {
        key: compute_engine.to_cpu(value) if hasattr(value, "shape") else value
        for key, value in arrays.items()
    }
    atomic_write_npz(trace_path, **arrays)

    roi_final = np.asarray(roi_psi_integrals[-1], dtype=np.float64) if roi_psi_integrals else np.zeros(len(masks))
    roi_peak = np.max(np.asarray(roi_psi_integrals, dtype=np.float64), axis=0) if roi_psi_integrals else np.zeros(len(masks))
    metadata = {
        "scenario": scenario,
        "trace_file": os.path.relpath(trace_path, run_dir),
        "elapsed_s": round(elapsed, 2),
        "n_steps": n_steps,
        "sample_frames": len(times),
        "dt_ps": dt,
        "scalar_stride_steps": stride,
        "condensed": condensed,
        "t_cond_ps": t_cond,
        "psi_sq_max_final": psi_sq_max[-1] if psi_sq_max else None,
        "psi_sq_max_peak": max(psi_sq_max) if psi_sq_max else None,
        "nR_max_peak": max(nR_max) if nR_max else None,
        "nI_max_peak": max(nI_max) if nI_max else None,
        "pump_dose": dose,
        "roi_final_integrals": {
            scenario["rois"][i]["id"]: float(roi_final[i]) for i in range(len(masks))
        },
        "roi_peak_integrals": {
            scenario["rois"][i]["id"]: float(roi_peak[i]) for i in range(len(masks))
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(
        os.path.join(run_dir, "metadata", f"{scenario['name']}.json"),
        metadata,
    )
    return metadata
