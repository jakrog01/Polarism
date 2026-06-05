"""GPU stage: WTA batch simulation — one task per batch of images.

Invoked as:
    python -m mnist_wta_v1.stages.gpu.run_batch \\
        --run-dir <run_dir> [--batch-index <int>] [--radius <float>]

--radius overrides geometry.ring_radius_um from config; must match the radius used
during calibrate_wta so P_th and class-spot positions are consistent.

SLURM_ARRAY_TASK_ID is used when --batch-index is not given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from mnist_wta_v1.config.loader import get_amplitude_cfg, get_encoding_cfg, get_geometry_cfg, get_readout_cfg
from mnist_wta_v1.encoding.class_geometry import class_spot_positions
from mnist_wta_v1.encoding.score_encoding import scores_to_powers
from mnist_wta_v1.simulation.batch_core import SharedSimResources, simulate_one_image_wta
from mnist_wta_v1.simulation.class_lasers import build_class_lasers
from mnist_common.io.atomic import atomic_write_json, atomic_write_npz
from mnist_common.simulation.roi import build_class_roi_masks
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


def _build_config(cfg: dict) -> Config:
    g = cfg["global"]["grid"]
    p = cfg["global"]["physics"]
    s = cfg["global"]["solver"]
    bc = cfg["global"].get("boundary_condition", {})
    r = cfg["global"].get("reservoir", {})
    return Config(
        grid=GridParameters(
            nx=int(g["nx"]), ny=int(g["ny"]),
            lx=float(g["lx"]), ly=float(g["ly"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="WTA GPU batch simulation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Ring radius in μm; overrides geometry.ring_radius_um in config.")
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
            print("ERROR: --batch-index not given and SLURM_ARRAY_TASK_ID not set.", file=sys.stderr)
            sys.exit(1)
        batch_idx = int(slurm_task)

    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    with open(os.path.join(run_dir, "batch_index.json")) as f:
        batch_index = json.load(f)

    if batch_idx >= len(batch_index):
        print(f"ERROR: batch_index={batch_idx} >= {len(batch_index)}", file=sys.stderr)
        sys.exit(1)

    batch_entry = batch_index[batch_idx]
    image_ids_in_batch = batch_entry["image_ids"]

    with open(os.path.join(run_dir, "dataset_index.json")) as f:
        dataset_index = json.load(f)
    id_to_entry = {e["image_id"]: e for e in dataset_index}

    calib_path = os.path.join(run_dir, "calibration_wta.json")
    if not os.path.isfile(calib_path):
        print("ERROR: calibration_wta.json not found — run calibrate_wta first.", file=sys.stderr)
        sys.exit(1)
    with open(calib_path) as f:
        calib_result = json.load(f)

    enc_cfg = get_encoding_cfg(cfg)
    readout_cfg = get_readout_cfg(cfg)
    amp_cfg = get_amplitude_cfg(cfg)
    geom_cfg = get_geometry_cfg(cfg)

    alpha = float(amp_cfg["alpha"])
    beta = float(amp_cfg["beta"])
    ring_r = args.radius if args.radius is not None else float(geom_cfg["ring_radius_um"])

    calib_radius = float(calib_result.get("radius_um", ring_r))
    if abs(calib_radius - ring_r) > 0.01:
        print(
            f"ERROR: calibration_wta.json was for R={calib_radius} μm "
            f"but run_batch is using R={ring_r} μm. "
            "Re-run calibrate_wta with the correct --radius.",
            file=sys.stderr,
        )
        sys.exit(1)

    P_th = float(calib_result["P_th_single_spot"])
    threshold_cond = float(calib_result["threshold_cond"])
    sigma_space = float(enc_cfg["sigma_space_um"])
    sigma_time = float(enc_cfg["sigma_time_ps"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])
    pulse_timing_mode = str(enc_cfg.get("pulse_timing_mode", "simultaneous"))
    min_gap = float(enc_cfg.get("min_inter_pulse_gap_ps", 0.0))
    readout_start = float(readout_cfg["window_start_ps"])
    readout_end = float(readout_cfg["window_end_ps"])
    readout_stride = int(readout_cfg.get("stride_steps", 100))
    roi_r = float(readout_cfg["roi_radius_um"])

    print("=" * 60)
    print(f" mnist_wta_v1 GPU batch {batch_idx} ({len(image_ids_in_batch)} images)")
    print("=" * 60)
    print(f"  R={ring_r} μm  alpha={alpha}  beta={beta}  P_th={P_th:.0f}  thr_cond={threshold_cond:.3e}")

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = _build_config(cfg)
    resources = SharedSimResources(sim_cfg)
    grid = resources.grid
    xp = compute_engine.xp

    xs_um, ys_um = class_spot_positions(ring_r)
    class_masks = build_class_roi_masks(xs_um, ys_um, roi_r, grid.X, grid.Y, xp)

    roi_features_list: list[np.ndarray] = []
    scalars_list: list[dict[str, Any]] = []
    image_ids_done: list[str] = []

    t_batch_start = time.monotonic()
    for local_i, image_id in enumerate(image_ids_in_batch):
        entry = id_to_entry[image_id]
        norm_scores = np.array(entry["norm_scores_per_class"], dtype=np.float64)
        label = int(entry["label"])

        powers = scores_to_powers(norm_scores, alpha, beta, P_th)
        lasers = build_class_lasers(
            xs_um=xs_um,
            ys_um=ys_um,
            powers=powers,
            sigma_space_um=sigma_space,
            sigma_time_ps=sigma_time,
            cutoff_sigma=cutoff,
            power_definition=power_def,
            grid_X=grid.X,
            grid_Y=grid.Y,
            pulse_timing_mode=pulse_timing_mode,
            min_inter_pulse_gap_ps=min_gap,
        )

        t_img_start = time.monotonic()
        try:
            result = simulate_one_image_wta(
                resources=resources,
                lasers=lasers,
                class_masks=class_masks,
                threshold_cond=threshold_cond,
                readout_window_start_ps=readout_start,
                readout_window_end_ps=readout_end,
                readout_stride_steps=readout_stride,
                power_per_class=powers.tolist(),
            )
        except Exception as exc:
            print(f"  ERROR on {image_id}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_cls = len(xs_um)
            result = {
                "class_roi_integrals": [0.0] * n_cls,
                "winner": -1,
                "margin": 0.0,
                "no_cond": True,
                "multi_cond": False,
                "power_per_class": powers.tolist(),
                "t_cond": None,
                "condensed": False,
                "readout_frames": 0,
            }

        elapsed_img = time.monotonic() - t_img_start
        roi_features_list.append(np.array(result["class_roi_integrals"], dtype=np.float64))
        scalars_list.append({
            "image_id": image_id,
            "label": label,
            "split": entry["split"],
            "winner": result["winner"],
            "margin": result["margin"],
            "no_cond": result["no_cond"],
            "multi_cond": result["multi_cond"],
            "class_roi_integrals": result["class_roi_integrals"],
            "power_per_class": result["power_per_class"],
            "condensed": result["condensed"],
            "t_cond": result["t_cond"],
            "readout_frames": result["readout_frames"],
            "correct": int(result["winner"] == label),
            "elapsed_s": round(elapsed_img, 2),
        })
        image_ids_done.append(image_id)

        print(
            f"  [{local_i + 1}/{len(image_ids_in_batch)}] {image_id}  "
            f"label={label}  winner={result['winner']}  "
            f"correct={result['winner'] == label}  "
            f"margin={result['margin']:.2f}  t={elapsed_img:.1f}s"
        )

    elapsed_batch = time.monotonic() - t_batch_start
    batch_name = f"batch_{batch_idx:04d}"

    atomic_write_npz(
        os.path.join(run_dir, "features", f"{batch_name}.npz"),
        roi_features=np.stack(roi_features_list, axis=0),
        image_ids=np.array(image_ids_done),
        labels=np.array([s["label"] for s in scalars_list], dtype=np.int64),
        splits=np.array([s["split"] for s in scalars_list]),
        winners=np.array([s["winner"] for s in scalars_list], dtype=np.int64),
    )

    atomic_write_json(
        os.path.join(run_dir, "metadata", f"{batch_name}.json"),
        {
            "batch_id": batch_idx,
            "image_ids": image_ids_done,
            "scalars": scalars_list,
            "amplitude_config": {"alpha": alpha, "beta": beta, "P_th": P_th},
            "geometry": {"ring_radius_um": ring_r, "roi_radius_um": roi_r},
            "pulse_config": {
                "laser_type": "pulse-gaussian",
                "sigma_time_ps": sigma_time,
                "cutoff_sigma": cutoff,
                "power_definition": power_def,
                "pulse_timing_mode": pulse_timing_mode,
                "min_inter_pulse_gap_ps": min_gap,
            },
            "threshold_cond": threshold_cond,
            "readout_config": {"window_start_ps": readout_start, "window_end_ps": readout_end},
            "elapsed_batch_s": round(elapsed_batch, 2),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )

    n_correct = sum(1 for s in scalars_list if s["correct"])
    n_cond = sum(1 for s in scalars_list if s["condensed"])
    print(
        f"\n  Batch {batch_idx}: {len(image_ids_done)} images  "
        f"correct={n_correct}/{len(image_ids_done)}  condensed={n_cond}  {elapsed_batch:.1f}s"
    )


if __name__ == "__main__":
    main()
