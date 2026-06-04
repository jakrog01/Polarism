"""GPU stage: simulate one batch of MNIST images sequentially on one GPU.

Slurm array: one task = one batch (N images).  Grid, BC, potential, solver
are shared across all images in the batch; psi and reservoir are reset per image.

Invoked as:
    python -m mnist_ring_v1.stages.gpu.run_batch \\
        --run-dir <run_dir> [--batch-index <int>]

SLURM_ARRAY_TASK_ID is used when --batch-index is not given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np

from mnist_ring_v1.config.loader import (
    get_encoding_cfg,
    get_grid_cfg,
    get_physics_cfg,
    get_readout_cfg,
)
from mnist_ring_v1.encoding.geometry import ring_spot_positions
from mnist_ring_v1.simulation.batch_core import SharedSimResources, simulate_one_image
from mnist_ring_v1.simulation.lasers import build_ring_trigger_lasers
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


def _atomic_write_npz(path: str, **arrays: np.ndarray) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, data: Any) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _build_config(cfg: dict) -> Config:
    g = get_grid_cfg(cfg)
    p = get_physics_cfg(cfg)
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
            mask_width_percent=float(bc.get("mask_width_percent", 0.2)),
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
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def _extract_threshold(calib_result: dict, mode_key: str) -> float | None:
    """Return estimated_threshold from a calibration result dict, or None if bracket not found."""
    entry = calib_result.get(mode_key, {})
    th = entry.get("estimated_threshold")
    if th is None:
        return None
    bracket = entry.get("bracket_found", True)
    if not bracket:
        return None
    return float(th)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU batch simulation for mnist_ring_v1"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-index", type=int, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.batch_index is not None:
        batch_idx = args.batch_index
    else:
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task is None:
            print(
                "ERROR: --batch-index not given and SLURM_ARRAY_TASK_ID not set.",
                file=sys.stderr,
            )
            sys.exit(1)
        batch_idx = int(slurm_task)

    with open(os.path.join(run_dir, "config.yaml")) as f:
        import yaml

        cfg = yaml.safe_load(f)

    with open(os.path.join(run_dir, "batch_index.json")) as f:
        batch_index = json.load(f)

    if batch_idx >= len(batch_index):
        print(
            f"ERROR: batch_index={batch_idx} out of range ({len(batch_index)} batches)",
            file=sys.stderr,
        )
        sys.exit(1)

    batch_entry = batch_index[batch_idx]
    image_ids_in_batch = batch_entry["image_ids"]

    with open(os.path.join(run_dir, "dataset_index.json")) as f:
        dataset_index = json.load(f)
    id_to_entry = {e["image_id"]: e for e in dataset_index}

    print("=" * 60)
    print(f" mnist_ring_v1 GPU batch {batch_idx} ({len(image_ids_in_batch)} images)")
    print("=" * 60)

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = _build_config(cfg)

    print("  Building shared GPU resources ...")
    resources = SharedSimResources(sim_cfg)
    grid = resources.grid

    enc_cfg = get_encoding_cfg(cfg)
    readout_cfg = get_readout_cfg(cfg)

    ring_xs, ring_ys = ring_spot_positions(
        float(enc_cfg["ring_radius_um"]),
        float(enc_cfg["ring_rotation_rad"]),
    )

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)

    calib_cfg = cfg.get("calibration", {})
    ring_sub_frac_config = float(calib_cfg.get("P_ring_subthreshold_fraction", 0.8))
    trig_frac = float(calib_cfg.get("P_trigger_fraction", 0.8))

    calib_result_path = os.path.join(run_dir, "calibration_result.json")
    if os.path.isfile(calib_result_path):
        with open(calib_result_path) as f:
            calib_result = json.load(f)
        ring_th = _extract_threshold(calib_result, "ring_only")
        trig_th = _extract_threshold(calib_result, "trigger_only")
        if ring_th is None or trig_th is None:
            print(
                "ERROR: calibration_result.json exists but threshold bracket not found "
                f"(ring_th={ring_th}, trig_th={trig_th}). "
                "Re-run calibration with a wider power sweep.",
                file=sys.stderr,
            )
            sys.exit(1)
        power_source = "calibration_result.json"
        ring_bracket_found = bool(
            calib_result.get("ring_only", {}).get("bracket_found", False)
        )
        trig_bracket_found = bool(
            calib_result.get("trigger_only", {}).get("bracket_found", False)
        )

        guard = calib_result.get("pretrigger_guard", {})
        guard_skipped = guard.get("skipped", True)
        guard_recommended_frac = guard.get("recommended_ring_fraction")

        if not guard_skipped and guard_recommended_frac is not None:
            ring_sub_frac = float(guard_recommended_frac)
            ring_frac_source = "pretrigger_guard"
            print(
                f"  Thresholds from calibration: P_ring={ring_th:.1f}, P_trigger={trig_th:.1f}"
            )
            print(
                f"  Ring fraction: {ring_sub_frac:.2f} (from pretrigger_guard; "
                f"config default was {ring_sub_frac_config:.2f})"
            )
        else:
            ring_sub_frac = ring_sub_frac_config
            ring_frac_source = "config_default" if guard_skipped else "config_default_guard_no_result"
            print(
                f"  Thresholds from calibration: P_ring={ring_th:.1f}, P_trigger={trig_th:.1f}"
            )
            if not guard_skipped:
                print(
                    f"  WARNING: pretrigger_guard has no recommended fraction; "
                    f"using config default {ring_sub_frac:.2f}"
                )
    else:
        ring_th = calib_cfg.get("P_ring_threshold")
        trig_th = calib_cfg.get("P_trigger_threshold")
        if ring_th is None or trig_th is None:
            print(
                "ERROR: No calibration_result.json and config.calibration.P_ring_threshold/"
                "P_trigger_threshold not set. Run calibration first.",
                file=sys.stderr,
            )
            sys.exit(1)
        ring_th = float(ring_th)
        trig_th = float(trig_th)
        power_source = "config.yaml"
        ring_bracket_found = None
        trig_bracket_found = None
        ring_sub_frac = ring_sub_frac_config
        ring_frac_source = "config_default"
        guard_recommended_frac = None
        print(
            f"  Thresholds from config: P_ring={ring_th:.1f}, P_trigger={trig_th:.1f}"
        )

    ring_power = ring_th * ring_sub_frac
    trigger_power = trig_th * trig_frac
    print(
        f"  ring_power={ring_power:.1f} ({ring_sub_frac}×P_ring, frac_source={ring_frac_source}), "
        f"trigger_power={trigger_power:.1f} ({trig_frac}×P_trig)"
    )

    T_max = float(enc_cfg["T_max"])
    readout_start = float(readout_cfg["window_start_offset_ps"]) + T_max
    readout_end = float(readout_cfg["window_end_offset_ps"]) + T_max
    kspace_crop = int(readout_cfg["kspace_crop_size"])
    kspace_norm = str(readout_cfg["normalization"])
    readout_stride = int(readout_cfg.get("stride_steps", 100))
    central_roi_um = float(readout_cfg.get("central_roi_um", 6.0))

    sigma_space = float(enc_cfg["sigma_space_um"])
    sigma_time = float(enc_cfg["sigma_time_ps"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])

    kspace_features: list[np.ndarray] = []
    scalars_list: list[dict] = []
    image_ids_done: list[str] = []

    t_batch_start = time.monotonic()
    for local_i, image_id in enumerate(image_ids_in_batch):
        entry = id_to_entry[image_id]
        ring_delays = np.array(entry["ring_delays_ps"], dtype=np.float64)
        trig_delay_val = float(entry["trigger_delay_ps"])

        lasers = build_ring_trigger_lasers(
            ring_xs=ring_xs,
            ring_ys=ring_ys,
            ring_delays=ring_delays,
            ring_power=ring_power,
            trigger_delay=trig_delay_val,
            trigger_power=trigger_power,
            sigma_space_um=sigma_space,
            sigma_time_ps=sigma_time,
            cutoff_sigma=cutoff,
            power_definition=power_def,
            grid_X=grid.X,
            grid_Y=grid.Y,
        )

        t_img_start = time.monotonic()
        try:
            result = simulate_one_image(
                resources=resources,
                lasers=lasers,
                readout_window_start_ps=readout_start,
                readout_window_end_ps=readout_end,
                kspace_crop_size=kspace_crop,
                kspace_normalization=kspace_norm,
                readout_stride_steps=readout_stride,
                trigger_delay_ps=trig_delay_val,
                central_roi_um=central_roi_um,
            )
        except Exception as exc:
            print(f"  ERROR on {image_id}: {exc}", file=sys.stderr)
            traceback.print_exc()
            result = {
                "kspace_feature": np.zeros(kspace_crop**2),
                "psi_sq_max": 0.0,
                "k0_frac": 0.0,
                "k_peak_um": 0.0,
                "k_centroid_um": 0.0,
                "high_k_frac_0p8_nyq": 0.0,
                "t_cond": None,
                "condensed": False,
                "readout_frames": 0,
                "psi_sq_central_max": 0.0,
                "central_condensed": False,
                "t_cond_central": None,
                "pretrigger_central_condensed": False,
                "detector_config": {},
            }

        elapsed_img = time.monotonic() - t_img_start
        kspace_features.append(result["kspace_feature"])
        scalars_list.append(
            {
                "image_id": image_id,
                "label": entry["label"],
                "split": entry["split"],
                "trigger_delay_ps": trig_delay_val,
                "T_max": T_max,
                "condensed": result["condensed"],
                "t_cond": result["t_cond"],
                "psi_sq_max": result["psi_sq_max"],
                "central_condensed": result.get("central_condensed", False),
                "t_cond_central": result.get("t_cond_central"),
                "psi_sq_central_max": result.get("psi_sq_central_max", 0.0),
                "pretrigger_central_condensed": result.get("pretrigger_central_condensed", False),
                "detector_config": result.get("detector_config", {}),
                "k0_frac": result["k0_frac"],
                "k_peak_um": result["k_peak_um"],
                "k_centroid_um": result["k_centroid_um"],
                "high_k_frac_0p8_nyq": result["high_k_frac_0p8_nyq"],
                "readout_frames": result["readout_frames"],
                "elapsed_s": round(elapsed_img, 2),
            }
        )
        image_ids_done.append(image_id)

        print(
            f"  [{local_i + 1}/{len(image_ids_in_batch)}] {image_id}  "
            f"condensed={result['condensed']}  "
            f"psi_sq_max={result['psi_sq_max']:.3e}  "
            f"t={elapsed_img:.1f}s"
        )

    elapsed_batch = time.monotonic() - t_batch_start

    batch_name = f"batch_{batch_idx:04d}"
    feat_path = os.path.join(run_dir, "features", f"{batch_name}.npz")
    _atomic_write_npz(
        feat_path,
        kspace_features=np.stack(kspace_features, axis=0),
        image_ids=np.array(image_ids_done),
        labels=np.array([s["label"] for s in scalars_list], dtype=np.int64),
        splits=np.array([s["split"] for s in scalars_list]),
    )

    meta_path = os.path.join(run_dir, "metadata", f"{batch_name}.json")
    _atomic_write_json(
        meta_path,
        {
            "batch_id": batch_idx,
            "image_ids": image_ids_done,
            "scalars": scalars_list,
            "power_provenance": {
                "source": power_source,
                "P_ring_threshold": ring_th,
                "P_trigger_threshold": trig_th,
                "P_ring_subthreshold_fraction": ring_sub_frac,
                "P_ring_fraction_source": ring_frac_source,
                "P_ring_fraction_config_default": ring_sub_frac_config,
                "P_trigger_fraction": trig_frac,
                "ring_power_used": ring_power,
                "trigger_power_used": trigger_power,
                "ring_bracket_found": ring_bracket_found,
                "trigger_bracket_found": trig_bracket_found,
                "guard_recommended_fraction": guard_recommended_frac,
            },
            "readout_config": {
                "window_start_ps": readout_start,
                "window_end_ps": readout_end,
                "stride_steps": readout_stride,
                "kspace_crop_size": kspace_crop,
                "normalization": kspace_norm,
            },
            "elapsed_batch_s": round(elapsed_batch, 2),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )

    n_condensed = sum(1 for s in scalars_list if s["condensed"])
    print(
        f"\n  Batch {batch_idx} done: {len(image_ids_done)} images, "
        f"{n_condensed} condensed, {elapsed_batch:.1f}s total."
    )
    print(f"  features -> {feat_path}")
    print(f"  metadata -> {meta_path}")


if __name__ == "__main__":
    main()
