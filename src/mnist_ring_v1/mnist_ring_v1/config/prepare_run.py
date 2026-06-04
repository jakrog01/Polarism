"""Build run directory and all pre-simulation artifacts.

Invoked as:
    python -m mnist_ring_v1.config.prepare_run \\
        --config <config.yaml> --run-dir <run_dir>
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

from mnist_ring_v1.config.loader import (
    get_encoding_cfg,
    get_mnist_cfg,
    get_projection_cfg,
    get_slurm_cfg,
    load_config,
)
from mnist_ring_v1.encoding.geometry import ring_spot_positions
from mnist_ring_v1.encoding.mnist import load_mnist, select_balanced_subset
from mnist_ring_v1.encoding.projection import (
    PCAProjection,
    apply_normalizer,
    fit_train_normalizer,
)
from mnist_ring_v1.encoding.ttfs import features_to_delays, trigger_delay

_MANIFEST_FILE = "manifest.json"
_DATASET_INDEX_FILE = "dataset_index.json"
_BATCH_INDEX_FILE = "batch_index.json"
_PROJECTION_FILE = "projection_model.npz"
_NORMALIZER_FILE = "feature_normalizer.npz"


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


def _compute_baseline_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Ridge linear classifier on normalized PCA features; returns test accuracy."""
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr_s, y_train)
        return float(np.mean(clf.predict(X_te_s) == y_test))
    except ImportError:
        pass

    mean = X_train.mean(axis=0)
    std = np.maximum(X_train.std(axis=0), 1e-8)
    X_tr_s = (X_train - mean) / std
    X_te_s = (X_test - mean) / std

    n_cls = 10
    Y_oh = np.eye(n_cls)[y_train]
    from scipy.linalg import lstsq
    X_tr_b = np.hstack([X_tr_s, np.ones((len(X_tr_s), 1))])
    W, _, _, _ = lstsq(X_tr_b, Y_oh)
    X_te_b = np.hstack([X_te_s, np.ones((len(X_te_s), 1))])
    y_pred = (X_te_b @ W).argmax(axis=1)
    return float(np.mean(y_pred == y_test))


def _save_pump_geometry(
    ring_xs: "np.ndarray",
    ring_ys: "np.ndarray",
    ring_r: float,
    sigma_space_um: float,
    ring_rotation_rad: float,
    out_path: str,
) -> None:
    """Save a diagnostic scatter plot of the ring+trigger pump geometry."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import numpy as _np
    except ImportError:
        print("  WARNING: matplotlib not available — pump_geometry.png skipped")
        return

    fwhm = 2.0 * _np.sqrt(2.0 * _np.log(2.0)) * sigma_space_um
    spot_r = fwhm / 2.0

    fig, ax = plt.subplots(figsize=(6, 6))
    for x, y in zip(ring_xs, ring_ys):
        circle = mpatches.Circle((x, y), spot_r, fill=True, alpha=0.35, color="steelblue", linewidth=0)
        ax.add_patch(circle)
        ax.plot(x, y, "+", color="steelblue", ms=6)

    trig = mpatches.Circle((0, 0), spot_r, fill=True, alpha=0.4, color="tomato", linewidth=0)
    ax.add_patch(trig)
    ax.plot(0, 0, "x", color="tomato", ms=8, mew=2)

    ring_outline = mpatches.Circle((0, 0), ring_r, fill=False, linestyle="--",
                                    edgecolor="gray", linewidth=0.8)
    ax.add_patch(ring_outline)

    ax.set_xlim(-ring_r * 1.4, ring_r * 1.4)
    ax.set_ylim(-ring_r * 1.4, ring_r * 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(
        f"Ring+trigger pump geometry\n"
        f"R={ring_r} μm  σ={sigma_space_um:.3f} μm  FWHM={fwhm:.2f} μm  "
        f"rot={_np.degrees(ring_rotation_rad):.1f}°"
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               alpha=0.7, markersize=10, label="ring spots (16)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               alpha=0.7, markersize=10, label="trigger (0,0)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  pump_geometry.png -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mnist_ring_v1 run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: balanced subset; overrides config n_train/n_test.",
    )
    parser.add_argument(
        "--pilot-n", type=int, default=None,
        help="Total images in pilot mode. Must be divisible by 10. "
             "Ignored if --pilot-train-per-class is set.",
    )
    parser.add_argument(
        "--pilot-train-per-class", type=int, default=None,
        help="Train images per class in pilot mode.",
    )
    parser.add_argument(
        "--pilot-test-per-class", type=int, default=None,
        help="Test images per class in pilot mode.",
    )
    parser.add_argument(
        "--pilot-mode",
        choices=["fast", "atlas", "accuracy", "custom"],
        default=None,
        help=(
            "Pilot preset: "
            "fast=5+5/class (100 imgs), "
            "atlas=25+5/class (300 imgs, response atlas + minimal test), "
            "accuracy=100+20/class (1200 imgs)."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mnist_cfg = get_mnist_cfg(cfg)
    proj_cfg = get_projection_cfg(cfg)
    enc_cfg = get_encoding_cfg(cfg)
    slurm_cfg = get_slurm_cfg(cfg)

    run_dir = os.path.abspath(args.run_dir)

    print("=" * 60)
    print(" mnist_ring_v1 — prepare_run")
    print("=" * 60)
    print(f"  config   : {args.config}")
    print(f"  run_dir  : {run_dir}")

    seed = int(mnist_cfg["seed"])

    if args.pilot or args.pilot_mode:
        presets = {
            "fast":     (5,   5),
            "atlas":    (25,  5),
            "accuracy": (100, 20),
        }
        # Priority (highest first):
        #  1. explicit --pilot-train-per-class / --pilot-test-per-class
        #  2. --pilot-n without --pilot-mode  (backward-compat: split evenly)
        #  3. --pilot-mode preset
        #  4. default "atlas" preset
        if args.pilot_train_per_class is not None or args.pilot_test_per_class is not None:
            tr_pc = args.pilot_train_per_class if args.pilot_train_per_class is not None else 25
            te_pc = args.pilot_test_per_class  if args.pilot_test_per_class  is not None else 0
            mode = "custom"
        elif args.pilot_n is not None and args.pilot_mode is None:
            pilot_n = args.pilot_n
            if pilot_n % 10 != 0:
                print(f"ERROR: --pilot-n={pilot_n} must be divisible by 10", file=sys.stderr)
                sys.exit(1)
            # Split evenly: half train half test, both rounded to multiples of 10
            half = pilot_n // 2
            tr_pc = half // 10
            te_pc = (pilot_n - tr_pc * 10) // 10
            mode = f"pilot_n_{pilot_n}"
        else:
            mode = args.pilot_mode or "atlas"
            tr_pc, te_pc = presets.get(mode, presets["atlas"])

        n_train = tr_pc * 10
        n_test = te_pc * 10
        atlas_only = te_pc == 0
        pilot_desc = f"PILOT/{mode}  {tr_pc}+{te_pc}/class  ({n_train} train + {n_test} test)"
        print(f"  mode     : {pilot_desc}")
        if atlas_only:
            print("  NOTE: atlas_only mode — no test split; all images go into train for response atlas")
        is_pilot = True
    else:
        n_train = int(mnist_cfg["n_train"])
        n_test = int(mnist_cfg["n_test"])
        is_pilot = False
        atlas_only = False

    print(f"  n_train  : {n_train}  n_test: {n_test}")

    print("\n[1/5] Loading MNIST ...")
    x_tr_all, y_tr_all, x_te_all, y_te_all = load_mnist(mnist_cfg["data_path"])
    print(f"  train pool: {len(x_tr_all)}, test pool: {len(x_te_all)}")

    try:
        tr_idx, x_train, y_train = select_balanced_subset(x_tr_all, y_tr_all, n_train, seed)
        if atlas_only or n_test == 0:
            te_idx = np.array([], dtype=np.int64)
            x_test = np.zeros((0, x_tr_all.shape[1]), dtype=np.float64)
            y_test = np.array([], dtype=np.int64)
        else:
            te_idx, x_test, y_test = select_balanced_subset(x_te_all, y_te_all, n_test, seed + 1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  selected: {len(x_train)} train, {len(x_test)} test  (balanced across 10 classes)")

    print("\n[2/5] Fitting PCA projection (train-only) ...")
    n_components = int(proj_cfg["n_components"])
    pca = PCAProjection(n_components=n_components)
    pca.fit(x_train)
    feats_train = pca.transform(x_train)
    feats_test = pca.transform(x_test)

    f_min, f_max = fit_train_normalizer(feats_train)
    feats_train_norm = apply_normalizer(feats_train, f_min, f_max)
    feats_test_norm = apply_normalizer(feats_test, f_min, f_max)

    print("\n[3/5] Baseline linear accuracy on 16 PCA features ...")
    if len(x_test) > 0:
        baseline_acc = _compute_baseline_accuracy(feats_train_norm, y_train, feats_test_norm, y_test)
        print(f"  Baseline (Ridge on {n_components} PCA features): {baseline_acc:.4f}")
    else:
        baseline_acc = None
        print("  Baseline skipped (atlas_only / no test split)")

    print("\n[4/5] Building dataset index ...")
    T_max = float(enc_cfg["T_max"])
    trigger_offset = T_max
    ring_r = float(enc_cfg["ring_radius_um"])
    ring_rot = float(enc_cfg["ring_rotation_rad"])
    ring_xs, ring_ys = ring_spot_positions(ring_r, ring_rot)

    dataset_entries: list[dict] = []
    for split, indices, images, labels_arr, feats_norm in (
        ("train", tr_idx, x_train, y_train, feats_train_norm),
        ("test", te_idx, x_test, y_test, feats_test_norm),
    ):
        for local_i in range(len(images)):
            delays = features_to_delays(feats_norm[local_i], T_max).tolist()
            trig_delay = trigger_delay(T_max)
            dataset_entries.append({
                "image_id": f"{split}_{local_i:05d}",
                "split": split,
                "dataset_index": int(indices[local_i]),
                "label": int(labels_arr[local_i]),
                "ring_delays_ps": delays,
                "trigger_delay_ps": trig_delay,
                "ring_xs_um": ring_xs.tolist(),
                "ring_ys_um": ring_ys.tolist(),
            })

    batch_size = int(slurm_cfg["batch_size"])
    n_total = len(dataset_entries)
    n_batches = math.ceil(n_total / batch_size)
    batches: list[list[str]] = []
    for b in range(n_batches):
        batch_ids = [
            dataset_entries[i]["image_id"]
            for i in range(b * batch_size, min((b + 1) * batch_size, n_total))
        ]
        batches.append(batch_ids)

    print(f"  total images: {n_total}, batches: {n_batches} (batch_size={batch_size})")

    if args.dry_run:
        print("\n[DRY RUN] Would create run directory and write:")
        print(f"  {run_dir}/{_MANIFEST_FILE}")
        print(f"  {run_dir}/{_DATASET_INDEX_FILE}  ({n_total} entries)")
        print(f"  {run_dir}/{_BATCH_INDEX_FILE}  ({n_batches} batches)")
        print(f"  {run_dir}/{_PROJECTION_FILE}")
        print(f"  {run_dir}/{_NORMALIZER_FILE}")
        print(f"\n  Baseline accuracy: {baseline_acc:.4f}")
        print(f"  n_steps per image: {int(cfg['global']['solver']['total_time'] / cfg['global']['solver']['dt']):,}")
        print(f"  readout window: T_max + {cfg['readout']['window_start_offset_ps']} .. + {cfg['readout']['window_end_offset_ps']} ps")
        print(f"  slurm tasks: {n_batches}")
        sys.exit(0)

    print("\n[5/5] Writing run directory ...")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    pca.save(os.path.join(run_dir, _PROJECTION_FILE))
    np.savez_compressed(
        os.path.join(run_dir, _NORMALIZER_FILE),
        f_min=f_min,
        f_max=f_max,
    )

    _save_pump_geometry(
        ring_xs=ring_xs,
        ring_ys=ring_ys,
        ring_r=ring_r,
        sigma_space_um=float(enc_cfg["sigma_space_um"]),
        ring_rotation_rad=float(enc_cfg["ring_rotation_rad"]),
        out_path=os.path.join(run_dir, "plots", "pump_geometry.png"),
    )

    manifest = {
        "config_path": args.config,
        "run_dir": run_dir,
        "pilot_mode": is_pilot,
        "pilot_atlas_only": atlas_only,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_total": n_total,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "baseline_pca_accuracy": baseline_acc,
        "prepare_complete": True,
        "gpu_complete": False,
        "finalize_complete": False,
    }
    _atomic_write_json(os.path.join(run_dir, _MANIFEST_FILE), manifest)

    _atomic_write_json(os.path.join(run_dir, _DATASET_INDEX_FILE), dataset_entries)
    _atomic_write_json(
        os.path.join(run_dir, _BATCH_INDEX_FILE),
        [{"batch_id": b, "image_ids": ids} for b, ids in enumerate(batches)],
    )

    print(f"  Manifest   : {os.path.join(run_dir, _MANIFEST_FILE)}")
    print(f"  Dataset    : {n_total} entries")
    print(f"  Batches    : {n_batches}")
    if baseline_acc is not None:
        print(f"  Baseline   : {baseline_acc:.4f}")
    print("\n  prepare_run complete.")


if __name__ == "__main__":
    main()
