"""GPU Stage 3: full 2-D Polarism simulation for one MNIST image.

Reads ``fit_result.json`` for the best ``sigma_space``, selects one image case
from ``reference/images/index.json`` by array task index, builds
``PulseGaussian`` lasers, runs the full spatial simulation, renders PNGs and
an MP4 animation (with MNIST thumbnail) while the HDF5 is still on local
scratch, generates a trace comparison plot against the ODE reference, then
persists only lightweight artifacts to the run directory.

Output per image::

  <run_dir>/results/images/<image_id>/scalars.npz
  <run_dir>/results/images/<image_id>/trace_comparison.png
  <run_dir>/results/images/<image_id>/diagnostic_overlay.png
  <run_dir>/results/images/<image_id>/image_meta.json
  <run_dir>/results/images/<image_id>/  (field PNGs, video)

Invoked as::

    python -m dot_response_fit.stages.gpu.run_scenario \\
        --run-dir <run_dir> [--image-index N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

from dot_response_fit.config.builder import build_mnist_lasers
from dot_response_fit.config.loader import extract_fit_cfg, load_config, make_dataclass
from dot_response_fit.manifest.io import (
    atomic_write_json,
    set_manifest_field,
)
from dot_response_fit.physics.trace_compare import score_traces
from dot_response_fit.stages.cpu.viz_engine import (
    FIELD_SPECS,
    generate_field_png,
    generate_trace_comparison,
    generate_diagnostic_overlay_png,
)
from dot_response_fit.stages.gpu.render_overlay import generate_animation_with_overlay
from pipeline.config.output_policy import output_policy_from_config
from pipeline.render.nvenc_stream import generate_animation
from pipeline.simulation.core import run_simulation_from_config
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


def _resolve_image_index(run_dir: str, index: int | None) -> int:
    if index is not None:
        return index
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task is None:
        print(
            "ERROR: --image-index not given and SLURM_ARRAY_TASK_ID not set.",
            file=sys.stderr,
        )
        sys.exit(1)
    return int(slurm_task)


def _make_scratch_dir(run_dir: str, image_id: str) -> str:
    run_name = os.path.basename(run_dir.rstrip("/"))
    job_id = (
        os.environ.get("POLARITON_SCRATCH_ID")
        or os.environ.get("SLURM_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or "local"
    )
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    unique_id = f"{job_id}_{task_id}"

    for env_var in ("SCRATCH", "SLURM_TMPDIR", "TMPDIR"):
        base = os.environ.get(env_var)
        if base and os.path.isdir(base):
            if os.path.realpath(base) == "/tmp":
                continue
            scratch = os.path.join(base, "polariton", run_name, unique_id, image_id)
            os.makedirs(scratch, exist_ok=False)
            return scratch

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError(
            "No suitable scratch directory found.  "
            "Set SCRATCH or SLURM_TMPDIR in the job environment."
        )
    return run_dir


def _render_artifacts(
    image_id: str,
    data_dir: str,
    scratch_results_dir: str,
    extent: list[float],
    render_snapshots: bool,
    render_animation: bool,
    render_overlay: bool,
    events_path: str,
    image_npy_path: str,
) -> None:
    if render_snapshots:
        for field_key in FIELD_SPECS:
            print(f"  PNG {field_key} ...", end="", flush=True)
            try:
                generate_field_png(
                    image_id, field_key, extent, data_dir, scratch_results_dir
                )
                print(" done")
            except Exception as e:
                print(f" WARNING: {e}", file=sys.stderr)

    if render_animation:
        print("  animation ...", flush=True)
        if render_overlay and os.path.isfile(events_path) and os.path.isfile(image_npy_path):
            try:
                generate_animation_with_overlay(
                    image_id,
                    FIELD_SPECS,
                    extent,
                    data_dir,
                    scratch_results_dir,
                    events_path=events_path,
                    image_path=image_npy_path,
                )
                return
            except Exception as e:
                print(
                    f"  WARNING: overlay animation failed ({e}), falling back to plain.",
                    file=sys.stderr,
                )
        generate_animation(image_id, FIELD_SPECS, extent, data_dir, scratch_results_dir)


def _collect_no_scratch_artifacts(
    run_dir: str,
    image_id: str,
    dst_image_dir: str,
    archive_raw_hdf5: bool,
) -> None:
    """Move artifacts from their default flat paths into *dst_image_dir*.

    When scratch is not used, run_simulation_from_config and _render_artifacts
    write to ``run_dir/results/<image_id>/`` and ``run_dir/<image_id>_*.npz``.
    This moves everything to ``results/images/<image_id>/`` so that finalize
    finds the same layout as the scratch-mode copyback produces.
    """
    os.makedirs(dst_image_dir, exist_ok=True)

    src_results = os.path.join(run_dir, "results", image_id)
    if os.path.isdir(src_results) and os.path.realpath(src_results) != os.path.realpath(dst_image_dir):
        for name in os.listdir(src_results):
            shutil.move(os.path.join(src_results, name), os.path.join(dst_image_dir, name))
        try:
            os.rmdir(src_results)
        except OSError:
            pass

    src_sidecar = os.path.join(run_dir, f"{image_id}_scalars.npz")
    if os.path.isfile(src_sidecar):
        shutil.move(src_sidecar, os.path.join(dst_image_dir, "scalars.npz"))

    src_h5 = os.path.join(run_dir, f"{image_id}.h5")
    if os.path.isfile(src_h5):
        if archive_raw_hdf5:
            shutil.move(src_h5, os.path.join(dst_image_dir, f"{image_id}.h5"))
        else:
            os.remove(src_h5)


def _copyback_artifacts(
    scratch_dir: str,
    run_dir: str,
    image_id: str,
    dst_image_dir: str,
    archive_raw_hdf5: bool,
) -> float:
    t0 = time.monotonic()

    src_results = os.path.join(scratch_dir, "results", image_id)
    if os.path.isdir(src_results):
        os.makedirs(os.path.dirname(dst_image_dir), exist_ok=True)
        if os.path.isdir(dst_image_dir):
            shutil.rmtree(dst_image_dir)
        shutil.copytree(src_results, dst_image_dir)

    src_sidecar = os.path.join(scratch_dir, f"{image_id}_scalars.npz")
    if os.path.isfile(src_sidecar):
        os.makedirs(dst_image_dir, exist_ok=True)
        shutil.copy2(src_sidecar, os.path.join(dst_image_dir, "scalars.npz"))

    if archive_raw_hdf5:
        src_h5 = os.path.join(scratch_dir, f"{image_id}.h5")
        if os.path.isfile(src_h5):
            dst_h5 = os.path.join(dst_image_dir, f"{image_id}.h5")
            tmp_dst = dst_h5 + ".tmp"
            try:
                shutil.copy2(src_h5, tmp_dst)
                os.replace(tmp_dst, dst_h5)
            except Exception:
                traceback.print_exc()
                if os.path.isfile(tmp_dst):
                    try:
                        os.remove(tmp_dst)
                    except OSError:
                        pass
                raise RuntimeError(f"HDF5 archive copy failed: {src_h5} -> {dst_h5}")

    return time.monotonic() - t0


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU Stage 3: full scenario simulation + render")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image-index", type=int, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    fit_path = os.path.join(run_dir, "fit_result.json")
    index_path = os.path.join(run_dir, "reference", "images", "index.json")

    for path in (config_path, fit_path, index_path):
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            sys.exit(1)

    image_idx = _resolve_image_index(run_dir, args.image_index)

    with open(index_path) as f:
        image_index: list[dict] = json.load(f)

    if image_idx >= len(image_index):
        print(
            f"ERROR: image_index={image_idx} out of range (have {len(image_index)} images)",
            file=sys.stderr,
        )
        sys.exit(1)

    image_entry = image_index[image_idx]
    image_id = image_entry["image_id"]
    events_path = os.path.join(run_dir, image_entry["paths"]["encoded_events"])
    target_trace_path = os.path.join(run_dir, image_entry["paths"]["target_trace"])
    image_npy_path = os.path.join(run_dir, image_entry["paths"]["input_image_npy"])

    cfg = load_config(config_path)
    global_cfg = cfg["global"]

    with open(fit_path) as f:
        fit_result: dict = json.load(f)

    with open(events_path) as f:
        encoded_events: dict = json.load(f)

    if not fit_result.get("search_completed"):
        print("ERROR: fit stage did not complete successfully.", file=sys.stderr)
        sys.exit(1)

    output_policy = output_policy_from_config(cfg)
    render_overlay = bool(cfg.get("output", {}).get("render_mnist_overlay", True))
    observable: str = fit_result.get("observable", "psi_sq_max")

    best_sigma_space: float = float(fit_result["best_sigma_space"])
    total_sim_time: float = float(encoded_events["total_sim_time"])

    grid_cfg = global_cfg.get("grid", {})
    lx = float(grid_cfg.get("lx", 250.0))
    ly = float(grid_cfg.get("ly", 250.0))
    extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

    print("=" * 60)
    print(f" GPU Stage 3: Image '{image_id}'  (index {image_idx})")
    print("=" * 60)
    print(f"  Run dir        : {run_dir}")
    print(f"  sigma_space    : {best_sigma_space:.2f} µm  (from fit)")
    print(f"  Fit score      : {fit_result.get('best_score', 'N/A')}")
    print(f"  n_pixels       : {encoded_events['n_pixels']}")
    print(f"  total_sim_time : {total_sim_time:.1f} ps")
    print(f"  dataset_index  : {image_entry['dataset_index']}  class={image_entry['digit_class']}")
    print(f"  archive_h5     : {output_policy.archive_raw_hdf5}")
    print(
        f"  SLURM job      : "
        f"{os.environ.get('SLURM_JOB_ID', 'N/A')} / "
        f"task {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}"
    )

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    physics = make_dataclass(PhysicsConstants, global_cfg.get("physics", {}))
    base_dt = float(global_cfg.get("solver", {}).get("dt", 0.001))
    solver_method = global_cfg.get("solver", {}).get("method", "rk4-cuda")

    sim_cfg = Config(
        grid=make_dataclass(GridParameters, global_cfg.get("grid", {})),
        boundary_condition=make_dataclass(
            BoundaryConditionParameters, global_cfg.get("boundary_condition", {})
        ),
        potential=make_dataclass(PotentialParameters, {"potential_type": "zero"}),
        physics=physics,
        laser=LaserParameters(),
        reservoir=make_dataclass(ReservoirParameters, global_cfg.get("reservoir", {})),
        solver=SolverParameters(
            total_time=total_sim_time,
            dt=base_dt,
            method=solver_method,
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )

    grid = create_grid(sim_cfg.grid)
    lasers = build_mnist_lasers(encoded_events, best_sigma_space, global_cfg, grid)
    laser_defaults = global_cfg.get("laser_defaults", {})
    fixed_laser_position = (
        float(laser_defaults.get("x0", 0.0)),
        float(laser_defaults.get("y0", 0.0)),
    )
    encoded_events.setdefault("laser_x0", fixed_laser_position[0])
    encoded_events.setdefault("laser_y0", fixed_laser_position[1])
    encoded_events.setdefault("spatial_encoding", "fixed-dot")

    n_steps = int(total_sim_time / base_dt)
    print(f"\n  Lasers : {len(lasers)}")
    print(
        f"  Pump xy: ({fixed_laser_position[0]:.3f}, "
        f"{fixed_laser_position[1]:.3f}) µm"
    )
    print(f"  Steps  : {n_steps:,}")

    try:
        scratch_dir = _make_scratch_dir(run_dir, image_id)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    using_scratch = scratch_dir != run_dir
    if using_scratch:
        print(f"  Scratch: {scratch_dir}")

    try:
        print("\n  Running simulation ...")
        t_sim_start = time.monotonic()
        t_cond, sidecar_path = run_simulation_from_config(
            image_id, lasers, sim_cfg, scratch_dir, output_policy
        )
        elapsed_sim = time.monotonic() - t_sim_start
    except Exception:
        traceback.print_exc()
        if using_scratch and os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)
        sys.exit(1)

    sidecar_data = np.load(sidecar_path) if sidecar_path and os.path.isfile(sidecar_path) else None
    t_sim = sidecar_data["time"] if sidecar_data is not None else np.empty(0)
    obs_sim = sidecar_data.get(observable, sidecar_data.get("psi_sq_max", np.empty(0))) if sidecar_data is not None else np.empty(0)

    target = np.load(target_trace_path)
    t_ref = target["time"]
    nC_ref = target["nC"]
    pump_ref = target.get("pump")

    rmse: float = float("inf")
    if t_sim.size > 0 and obs_sim.size > 0:
        rmse, _tc, _rn, _sn = score_traces(t_ref, nC_ref, t_sim, obs_sim)

    print(f"\n  Simulation complete.  t_cond={t_cond}  RMSE={rmse:.6f}")

    dst_image_dir = os.path.join(run_dir, "results", "images", image_id)

    scratch_results_dir = os.path.join(scratch_dir, "results")
    scratch_image_results_dir = os.path.join(scratch_results_dir, image_id)
    os.makedirs(scratch_image_results_dir, exist_ok=True)

    print("\n  Rendering artifacts from scratch ...")
    _render_artifacts(
        image_id,
        scratch_dir,
        scratch_results_dir,
        extent,
        output_policy.render_snapshots,
        output_policy.render_animation,
        render_overlay,
        events_path,
        image_npy_path,
    )

    if t_sim.size > 0 and obs_sim.size > 0:
        print("  trace_comparison.png ...", end="", flush=True)
        try:
            tc_path = generate_trace_comparison(
                image_id=image_id,
                t_ref=t_ref,
                nC_ref=nC_ref,
                pump_ref=pump_ref,
                t_sim=t_sim,
                obs_sim=obs_sim,
                sigma_space=best_sigma_space,
                rmse=rmse,
                dataset_index=image_entry["dataset_index"],
                digit_class=image_entry["digit_class"],
                observable_label=observable,
                out_dir=scratch_image_results_dir,
            )
            print(f" done  ({tc_path})")
        except Exception as exc:
            print(f" WARNING: {exc}", file=sys.stderr)

    h5_scratch_path = os.path.join(scratch_dir, f"{image_id}.h5")
    if os.path.isfile(h5_scratch_path) and os.path.isfile(image_npy_path):
        print("  diagnostic_overlay.png ...", end="", flush=True)
        try:
            with open(events_path) as f:
                events_dict = json.load(f)
            image_norm = np.load(image_npy_path).astype(np.float64)
            if image_norm.max() > 1.0:
                image_norm /= image_norm.max()
            do_path = generate_diagnostic_overlay_png(
                h5_path=h5_scratch_path,
                extent=extent,
                events=events_dict,
                image_norm=image_norm,
                out_dir=scratch_image_results_dir,
                laser_position=fixed_laser_position,
            )
            print(f" done  ({do_path})")
        except Exception as exc:
            print(f" WARNING: {exc}", file=sys.stderr)

    shutil.copy2(image_npy_path, os.path.join(scratch_image_results_dir, "input_image.npy"))
    png_src = os.path.join(run_dir, image_entry["paths"]["input_image_png"])
    if os.path.isfile(png_src):
        shutil.copy2(png_src, os.path.join(scratch_image_results_dir, "input_image.png"))

    copyback_seconds: float = 0.0
    if using_scratch:
        print(f"\n  Copying artifacts scratch → {run_dir} ...")
        try:
            copyback_seconds = _copyback_artifacts(
                scratch_dir, run_dir, image_id, dst_image_dir,
                output_policy.archive_raw_hdf5,
            )
        except Exception:
            traceback.print_exc()
            print(
                f"ERROR: artifact copy-back failed; preserving scratch: {scratch_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Copy-back done  ({copyback_seconds:.1f}s)")
        shutil.rmtree(scratch_dir, ignore_errors=True)
    else:
        _collect_no_scratch_artifacts(
            run_dir, image_id, dst_image_dir, output_policy.archive_raw_hdf5
        )

    h5_final = os.path.join(dst_image_dir, f"{image_id}.h5")
    h5_bytes = os.path.getsize(h5_final) if os.path.isfile(h5_final) else 0

    meta: dict = {
        "image_id": image_id,
        "image_index": image_idx,
        "dataset_index": image_entry["dataset_index"],
        "digit_class": image_entry["digit_class"],
        "n_pixels": encoded_events["n_pixels"],
        "spatial_encoding": "fixed-dot",
        "laser_x0": fixed_laser_position[0],
        "laser_y0": fixed_laser_position[1],
        "sigma_space": best_sigma_space,
        "fit_score": rmse if rmse != float("inf") else None,
        "observable": observable,
        "total_sim_time": total_sim_time,
        "dt": base_dt,
        "n_steps": n_steps,
        "t_cond": t_cond,
        "paths": {
            "scalars": "scalars.npz",
            "trace_comparison": "trace_comparison.png",
            "diagnostic_overlay": "diagnostic_overlay.png",
            "input_image_png": "input_image.png",
            "h5_file": f"{image_id}.h5" if output_policy.archive_raw_hdf5 else None,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "telemetry": {
            "elapsed_sim_seconds": round(elapsed_sim, 2),
            "n_steps": n_steps,
            "h5_bytes": h5_bytes,
            "copyback_seconds": round(copyback_seconds, 2),
        },
    }
    os.makedirs(dst_image_dir, exist_ok=True)
    meta_path = os.path.join(dst_image_dir, "image_meta.json")
    atomic_write_json(meta_path, meta)
    print(f"  Metadata: {meta_path}")
    print(f"\n  Stage 3 '{image_id}' complete.  t_cond={t_cond}  RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
