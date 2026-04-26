"""GPU Stage 2: fit Gaussian dot spatial size globally across all reference images.

Algorithm
---------
For each candidate *sigma_space* from ``fit.sigma_space_values``:

1. Iterate over all images from ``reference/images/index.json``.
2. For each image:
   a. Load ``encoded_events.json`` and ``target_trace.npz``.
   b. Build ``PulseGaussian`` lasers (all pixels, or first ``n_fit_pixels`` if set).
   c. If ``n_fit_pixels`` is null, the 2-D pump matches the ODE pump exactly —
      load ``target_trace.npz`` as the ODE reference.
      If ``n_fit_pixels`` is set, recompute a subset ODE reference so that ODE
      and 2-D pump see the same pulse sequence.
   d. Run a scalar-only simulation (no HDF5) and compute normalised RMSE.
3. Aggregate per-image RMSE: ``mean_rmse`` over finite scores.
4. Select the ``sigma_space`` with the lowest aggregate score.

Output: ``fit_result.json`` in the run directory.

Invoked as::

    python -m dot_response_fit.stages.gpu.fit_dot_size --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np

from dot_response_fit.config.builder import build_mnist_lasers
from dot_response_fit.config.loader import (
    extract_fit_cfg,
    load_config,
    make_dataclass,
)
from dot_response_fit.physics.reference_ode import params_from_physics, run_reference_ode
from dot_response_fit.physics.trace_compare import score_traces
from dot_response_fit.manifest.io import atomic_write_json, set_manifest_field
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    ComputeEngineParameters,
    PhysicsConstants,
    SolverParameters,
)
from polarism.grid.create_grid import create_grid
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver

RNG_SEED = 42
_CHECK_EVERY = 200
_BLOWUP_FACTOR = 1e8
_SUPPORTED_OBSERVABLES = {"psi_sq_max", "integrated_psi_sq"}


def _precompute_spatial_profiles(lasers: list, grid: object) -> list:
    return [laser._P_space(grid.X, grid.Y) for laser in lasers]


def _p_time_pure(
    t: float, pulse_separation: float, cutoff_sigma: float, sigma_time: float
) -> float:
    phase = cutoff_sigma * sigma_time
    n = max(0, round((t - phase) / pulse_separation))
    dt_val = t - n * pulse_separation - phase
    if abs(dt_val) > cutoff_sigma * sigma_time:
        return 0.0
    return math.exp(-0.5 * (dt_val / sigma_time) ** 2)


def _compute_pump(lasers: list, profiles: list, t: float, xp: object, P_zero: object) -> object:
    P_total = P_zero.copy()
    for i, laser in enumerate(lasers):
        t_eff = t - laser.delay
        if t_eff < 0:
            continue
        if laser.n_pulses > 0:
            n_pulse = max(0, round((t_eff - laser.phase) / laser.pulse_separation))
            if n_pulse >= laser.n_pulses:
                continue
        amp = laser._amplitude(t_eff)
        pt = _p_time_pure(t_eff, laser.pulse_separation, laser.cutoff_sigma, laser.sigma_time)
        temporal = float(amp) * pt
        if temporal > 0.0:
            P_total = P_total + temporal * profiles[i]
    return P_total


def _run_scalar_trace(
    lasers: list,
    cfg: object,
    dt: float,
    total_time: float,
    scalar_stride: int,
) -> dict[str, np.ndarray]:
    """Run a scalar-only 2-D Polarism simulation; return observable traces.

    Returns a dict with keys:
    * ``time``             — sample timestamps (ps)
    * ``psi_sq_max``       — peak spatial density at each sample
    * ``integrated_psi_sq``— cumulative integral of spatial norm
    """
    xp = compute_engine.xp

    grid = create_grid(cfg.grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics)
    potential = create_potential(cfg.potential, grid)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid)

    dx = float(grid.lx / grid.nx)
    dy = float(grid.ly / grid.ny)
    cell_area = dx * dy

    rng = xp.random.default_rng(RNG_SEED)
    state = SimulationState.__new__(SimulationState)
    state.psi = (
        cfg.physics.init_eps
        * (
            rng.random((grid.ny, grid.nx), dtype=xp.float64)
            + 1j * rng.random((grid.ny, grid.nx), dtype=xp.float64)
        )
    ).astype(xp.complex128)
    state.t = 0.0

    cfg.solver = SolverParameters(
        total_time=total_time, dt=dt, method=cfg.solver.method
    )
    solver = create_solver(cfg, grid)

    cap = bc.before_step_action()
    if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
        potential = potential.astype(state.psi.dtype)
        cap = cap.astype(state.psi.dtype)
    potential = potential + cap

    profiles = _precompute_spatial_profiles(lasers, grid)
    P_zero = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)

    n_steps = int(total_time / dt)
    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))
    times: list[float] = []
    psi_sq_maxes: list[float] = []
    integrated_values: list[float] = []
    running_integral: float = 0.0
    prev_t: float = 0.0

    for step in range(n_steps):
        t = step * dt
        P_total = _compute_pump(lasers, profiles, t, xp, P_zero)
        solver.step(potential, P_total, reservoir, bc, state)

        if step % scalar_stride == 0:
            psi_sq = xp.abs(state.psi) ** 2
            psi_sq_max = float(xp.max(psi_sq))
            spatial_norm = float(xp.sum(psi_sq)) * cell_area
            if step > 0:
                running_integral += spatial_norm * (t - prev_t)
            prev_t = t
            times.append(t)
            psi_sq_maxes.append(psi_sq_max)
            integrated_values.append(running_integral)

        if step % _CHECK_EVERY == 0 and step > 0:
            N_total = float(xp.sum(xp.abs(state.psi) ** 2))
            if math.isnan(N_total) or math.isinf(N_total):
                raise RuntimeError(f"NaN/Inf at step {step}, t={t:.3f} ps")
            if N_total > N_initial * _BLOWUP_FACTOR:
                raise RuntimeError(
                    f"Norm blow-up at step {step}, t={t:.3f} ps "
                    f"(N={N_total:.2e} vs N0={N_initial:.2e})"
                )

    return {
        "time": np.array(times, dtype=np.float64),
        "psi_sq_max": np.array(psi_sq_maxes, dtype=np.float64),
        "integrated_psi_sq": np.array(integrated_values, dtype=np.float64),
    }


def _build_fit_scenario_config(
    global_cfg: dict[str, Any],
    sigma_space: float,
    total_time: float,
    dt: float,
) -> object:
    from polarism.config.simulation_parameters import (
        BoundaryConditionParameters,
        Config,
        GridParameters,
        LaserParameters,
        PotentialParameters,
        ReservoirParameters,
        ResultParameters,
    )

    g = global_cfg
    physics = make_dataclass(PhysicsConstants, g.get("physics", {}))

    return Config(
        grid=make_dataclass(GridParameters, g.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, g.get("boundary_condition", {})
        ),
        potential=make_dataclass(PotentialParameters, {"potential_type": "zero"}),
        physics=physics,
        laser=LaserParameters(),
        reservoir=make_dataclass(ReservoirParameters, g.get("reservoir", {})),
        solver=SolverParameters(
            total_time=total_time,
            dt=dt,
            method=g.get("solver", {}).get("method", "rk4-cuda"),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def _score_image(
    encoded_events: dict[str, Any],
    target_trace_path: str,
    global_cfg: dict[str, Any],
    sigma_space: float,
    fit_dt: float,
    scalar_stride: int,
    observable: str,
    n_fit: int | None,
    ode_params: dict[str, float],
    n_ref_points: int,
    ode_rtol: float,
    ode_atol: float,
    grid: object,
) -> dict[str, Any]:
    """Score one image at a given sigma_space.

    Returns a dict with ``score``, ``finite``, ``n_pixels_used``, ``subset_ode``.
    """
    n_pixels = encoded_events["n_pixels"]
    pixel_limit = n_fit if (n_fit is not None and n_fit < n_pixels) else None
    n_active = pixel_limit if pixel_limit is not None else n_pixels

    amps_all = np.array(encoded_events["amplitudes"])
    centers_all = np.array(encoded_events["centers_ode"])
    sigma_t = float(encoded_events["sigma_time"])
    phase = float(encoded_events["phase"])

    if pixel_limit is not None:
        amps_ref = amps_all[:n_active]
        centers_ref = centers_all[:n_active]
        fit_total_time = float(centers_ref[-1]) + phase + 200.0
        t_ref, nC_ref, _ = run_reference_ode(
            amps_ref, centers_ref, sigma_t, ode_params,
            fit_total_time, n_ref_points, rtol=ode_rtol, atol=ode_atol,
        )
    else:
        target = np.load(target_trace_path)
        t_ref = target["time"]
        nC_ref = target["nC"]
        fit_total_time = float(encoded_events["total_sim_time"])

    lasers = build_mnist_lasers(
        encoded_events, sigma_space, global_cfg, grid, pixel_limit=pixel_limit
    )
    fit_cfg_obj = _build_fit_scenario_config(global_cfg, sigma_space, fit_total_time, fit_dt)
    traces = _run_scalar_trace(lasers, fit_cfg_obj, fit_dt, fit_total_time, scalar_stride)
    rmse, _tc, _rn, _sn = score_traces(t_ref, nC_ref, traces["time"], traces[observable])
    finite = math.isfinite(rmse)
    return {
        "score": rmse if finite else None,
        "finite": finite,
        "n_pixels_used": n_active,
        "subset_ode": pixel_limit is not None,
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU Stage 2: fit Gaussian dot size")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    index_path = os.path.join(run_dir, "reference", "images", "index.json")

    for path in (config_path, index_path):
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(config_path)
    global_cfg = cfg["global"]
    fit_cfg = extract_fit_cfg(cfg)

    observable: str = fit_cfg.get("observable", "psi_sq_max")
    if observable not in _SUPPORTED_OBSERVABLES:
        print(
            f"ERROR: fit.observable={observable!r} not supported. "
            f"Allowed: {sorted(_SUPPORTED_OBSERVABLES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(index_path) as f:
        image_index: list[dict] = json.load(f)

    sigma_space_values: list[float] = [float(v) for v in fit_cfg["sigma_space_values"]]
    max_runtime_minutes: float = float(fit_cfg.get("max_runtime_minutes", 120))
    max_wall = max_runtime_minutes * 60.0
    dt_factor: int = int(fit_cfg.get("dt_factor", 5))
    scalar_stride: int = int(fit_cfg.get("scalar_stride", 50))
    n_ref_points: int = int(fit_cfg.get("n_ref_points", 2000))

    n_fit_pixels_raw = fit_cfg.get("n_fit_pixels")
    n_fit: int | None = int(n_fit_pixels_raw) if n_fit_pixels_raw is not None else None

    base_dt = float(global_cfg.get("solver", {}).get("dt", 0.001))
    fit_dt = base_dt * dt_factor

    ref_section = cfg.get("reference", {})
    nc_source = float(ref_section.get("nc_source", 1.0e-6))
    physics = make_dataclass(PhysicsConstants, global_cfg.get("physics", {}))
    ode_params = params_from_physics(physics)
    ode_params["nc_source"] = nc_source
    ode_rtol = float(ref_section.get("ode_solver_rtol", 1e-6))
    ode_atol = float(ref_section.get("ode_solver_atol", 1e-7))

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    from polarism.config.simulation_parameters import GridParameters
    shared_grid = create_grid(make_dataclass(GridParameters, global_cfg.get("grid", {})))

    print("=" * 60)
    print(" GPU Stage 2: Fit Gaussian Dot Size")
    print("=" * 60)
    print(f"  Run dir           : {run_dir}")
    print(f"  observable        : {observable}")
    print(f"  sigma_space values: {sigma_space_values}")
    print(f"  n_images          : {len(image_index)}")
    print(f"  n_fit_pixels      : {'all' if n_fit is None else n_fit}")
    print(f"  fit_dt            : {fit_dt} ps  ({dt_factor}x coarser)")
    print(f"  Wall-clock cap    : {max_runtime_minutes} min")
    print()

    wall_start = time.time()
    all_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for sigma_space in sigma_space_values:
        elapsed = time.time() - wall_start
        if elapsed > max_wall:
            print(f"\n  Wall-clock limit reached ({elapsed:.0f}s)")
            break

        print(f"  sigma_space={sigma_space:.2f} µm", flush=True)
        per_image: list[dict[str, Any]] = []
        finite_scores: list[float] = []

        for entry in image_index:
            image_id = entry["image_id"]
            events_path = os.path.join(run_dir, entry["paths"]["encoded_events"])
            trace_path = os.path.join(run_dir, entry["paths"]["target_trace"])

            with open(events_path) as f:
                events = json.load(f)

            print(f"    {image_id} (dataset={entry['dataset_index']}, class={entry['digit_class']}) ...", end="", flush=True)
            try:
                result = _score_image(
                    encoded_events=events,
                    target_trace_path=trace_path,
                    global_cfg=global_cfg,
                    sigma_space=sigma_space,
                    fit_dt=fit_dt,
                    scalar_stride=scalar_stride,
                    observable=observable,
                    n_fit=n_fit,
                    ode_params=ode_params,
                    n_ref_points=n_ref_points,
                    ode_rtol=ode_rtol,
                    ode_atol=ode_atol,
                    grid=shared_grid,
                )
                failure_reason = None
            except Exception as exc:
                import traceback
                traceback.print_exc()
                result = {"score": None, "finite": False, "n_pixels_used": 0, "subset_ode": False}
                failure_reason = str(exc)

            score_val = result["score"]
            finite = result["finite"]
            print(f"  RMSE={score_val:.6f}" if finite else "  FAILED")

            image_rec: dict[str, Any] = {
                "image_id": image_id,
                "dataset_index": entry["dataset_index"],
                "digit_class": entry["digit_class"],
                "score": score_val,
                "finite": finite,
                "n_pixels_used": result.get("n_pixels_used"),
                "subset_ode": result.get("subset_ode"),
            }
            if failure_reason:
                image_rec["failure_reason"] = failure_reason
            per_image.append(image_rec)
            if finite:
                finite_scores.append(score_val)

        n_failed = len(per_image) - len(finite_scores)
        mean_rmse_diagnostic = float(np.mean(finite_scores)) if finite_scores else None
        eligible = (n_failed == 0) and (len(finite_scores) == len(per_image))

        if eligible:
            aggregate_score = float(finite_scores[0]) if len(finite_scores) == 1 else float(np.mean(finite_scores))
            print(
                f"    aggregate mean_rmse={aggregate_score:.6f}"
                f"  (all {len(per_image)} images OK)"
            )
        else:
            aggregate_score = float("inf")
            print(
                f"    INELIGIBLE: n_failed={n_failed}/{len(per_image)}"
                + (f"  diagnostic_mean_rmse={mean_rmse_diagnostic:.6f}" if mean_rmse_diagnostic is not None else "")
            )

        res: dict[str, Any] = {
            "sigma_space": float(sigma_space),
            "eligible": eligible,
            "aggregate_score": aggregate_score if eligible else None,
            "mean_rmse": mean_rmse_diagnostic,
            "max_rmse": float(max(finite_scores)) if finite_scores else None,
            "n_images": len(per_image),
            "n_failed_images": n_failed,
            "per_image": per_image,
        }
        all_results.append(res)

        if eligible and (
            best is None or aggregate_score < best["aggregate_score"]
        ):
            best = res

    output: dict[str, Any] = {
        "search_completed": best is not None,
        "observable": observable,
        "aggregate_method": "mean_rmse",
        "best_sigma_space": best["sigma_space"] if best else None,
        "best_score": best["aggregate_score"] if best else None,
        "n_images": len(image_index),
        "image_ids": [e["image_id"] for e in image_index],
        "n_fit_pixels": n_fit,
        "fit_dt": fit_dt,
        "fit_reference_source": "target_trace_npz" if n_fit is None else "subset_ode",
        "fit_reference_nc_source": nc_source,
        "fit_reference_n_points": n_ref_points,
        "all_results": all_results,
        "wall_time_seconds": round(time.time() - wall_start, 1),
    }

    if best is None:
        output["failure_reason"] = "no finite aggregate score for any sigma_space"
        print("\n  ERROR: all candidates failed — no finite score obtained.", file=sys.stderr)

    out_path = os.path.join(run_dir, "fit_result.json")
    atomic_write_json(out_path, output)
    print(f"\n  Saved: {out_path}")

    if best:
        print(
            f"  Best sigma_space: {best['sigma_space']:.2f} µm  "
            f"(mean_rmse={best['aggregate_score']:.6f})"
        )
        try:
            set_manifest_field(run_dir, "fit_complete", True)
            set_manifest_field(run_dir, "fit_result", {
                "best_sigma_space": best["sigma_space"],
                "best_score": best["aggregate_score"],
            })
        except Exception as exc:
            print(f"  WARNING: manifest update failed: {exc}", file=sys.stderr)
    else:
        try:
            set_manifest_field(run_dir, "fit_complete", False)
        except Exception as exc:
            print(f"  WARNING: manifest update failed: {exc}", file=sys.stderr)

    sys.exit(0 if best else 1)


if __name__ == "__main__":
    main()
