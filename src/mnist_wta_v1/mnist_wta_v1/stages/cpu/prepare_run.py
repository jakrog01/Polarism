"""Build WTA run directory: PCA + Ridge score model + dataset index with norm scores.

Invoked as:
    python -m mnist_wta_v1.stages.cpu.prepare_run \\
        --config <config.yaml> --run-dir <run_dir> [--dry-run]
        [--pilot] [--pilot-train-per-class N] [--pilot-test-per-class N]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from typing import Any

import numpy as np

from mnist_wta_v1.config.loader import (
    get_classifier_cfg,
    get_encoding_cfg,
    get_geometry_cfg,
    get_mnist_cfg,
    get_projection_cfg,
    get_slurm_cfg,
    load_config,
)
from mnist_wta_v1.encoding.class_geometry import class_spot_positions
from mnist_wta_v1.encoding.classifier_scores import (
    fit_score_model,
    predict_scores,
    save_score_model,
)
from mnist_wta_v1.encoding.score_encoding import normalize_scores_per_image
from mnist_common.io.atomic import atomic_write_json
from mnist_common.encoding.mnist import load_mnist, select_balanced_subset


def _pca_fit_transform(
    x_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on train, transform both splits. Returns train_feat, test_feat, mean, components."""
    mean = x_train.mean(axis=0)
    centered = x_train - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    train_feat = (x_train - mean) @ components.T
    test_feat = (x_test - mean) @ components.T if len(x_test) > 0 else np.zeros((0, n_components))
    return train_feat, test_feat, mean, components


def _minmax_normalize_features(
    feat_train: np.ndarray,
    feat_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_min = feat_train.min(axis=0)
    f_max = feat_train.max(axis=0)
    rng = np.where(f_max - f_min > 1e-8, f_max - f_min, 1.0)
    norm_train = (feat_train - f_min) / rng
    norm_test = (feat_test - f_min) / rng if len(feat_test) > 0 else feat_test
    return norm_train, norm_test, f_min, f_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mnist_wta_v1 run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-train-per-class", type=int, default=None)
    parser.add_argument("--pilot-test-per-class", type=int, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Ring radius in μm; overrides geometry.ring_radius_um for manifest logging.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mnist_cfg = get_mnist_cfg(cfg)
    proj_cfg = get_projection_cfg(cfg)
    clf_cfg = get_classifier_cfg(cfg)
    enc_cfg = get_encoding_cfg(cfg)
    geom_cfg = get_geometry_cfg(cfg)
    slurm_cfg = get_slurm_cfg(cfg)

    run_dir = os.path.abspath(args.run_dir)

    print("=" * 60)
    print(" mnist_wta_v1 — prepare_run")
    print("=" * 60)
    print(f"  config   : {args.config}")
    print(f"  run_dir  : {run_dir}")

    seed = int(mnist_cfg["seed"])

    if args.pilot or args.pilot_train_per_class is not None or args.pilot_test_per_class is not None:
        tr_pc = args.pilot_train_per_class if args.pilot_train_per_class is not None else 25
        te_pc = args.pilot_test_per_class if args.pilot_test_per_class is not None else 5
        n_train = tr_pc * 10
        n_test = te_pc * 10
        is_pilot = True
    else:
        n_train = int(mnist_cfg["n_train"])
        n_test = int(mnist_cfg["n_test"])
        is_pilot = False

    print(f"  n_train  : {n_train}  n_test: {n_test}{'  [PILOT]' if is_pilot else ''}")

    print("\n[1/6] Loading MNIST ...")
    x_tr_all, y_tr_all, x_te_all, y_te_all = load_mnist(mnist_cfg["data_path"])
    print(f"  pool: {len(x_tr_all)} train, {len(x_te_all)} test")

    try:
        tr_idx, x_train, y_train = select_balanced_subset(x_tr_all, y_tr_all, n_train, seed)
        te_idx, x_test, y_test = select_balanced_subset(x_te_all, y_te_all, n_test, seed + 1) if n_test > 0 else (np.array([], dtype=np.int64), np.zeros((0, 784)), np.array([], dtype=np.int64))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  selected: {len(x_train)} train, {len(x_test)} test")

    print("\n[2/6] Fitting PCA ...")
    n_components = int(proj_cfg["n_components"])
    feat_train, feat_test, pca_mean, pca_components = _pca_fit_transform(
        x_train, x_test, n_components
    )

    feat_train_norm, feat_test_norm, f_min, f_max = _minmax_normalize_features(
        feat_train, feat_test
    )

    print(f"\n[3/6] Fitting Ridge classifier (alpha={clf_cfg.get('alpha', 1.0)}) ...")
    score_model = fit_score_model(feat_train_norm, y_train, alpha=float(clf_cfg.get("alpha", 1.0)))
    print(f"  Ridge train accuracy: {score_model['train_accuracy']:.4f}")

    baseline_acc: float | None = None
    if len(x_test) > 0:
        scores_test = predict_scores(score_model, feat_test_norm)
        y_pred_test = score_model["classes"][np.argmax(scores_test, axis=1)]
        baseline_acc = float(np.mean(y_pred_test == y_test))
        print(f"  Ridge test accuracy (= baseline for WTA): {baseline_acc:.4f}")

    print("\n[4/6] Building dataset index ...")
    ring_r = args.radius if args.radius is not None else float(geom_cfg["ring_radius_um"])
    xs_um, ys_um = class_spot_positions(ring_r)

    dataset_entries: list[dict[str, Any]] = []
    for split, indices, images, labels_arr, feats_norm in (
        ("train", tr_idx, x_train, y_train, feat_train_norm),
        ("test", te_idx, x_test, y_test, feat_test_norm),
    ):
        if len(images) == 0:
            continue
        scores_split = predict_scores(score_model, feats_norm)
        for local_i in range(len(images)):
            raw_scores = scores_split[local_i]
            norm_scores = normalize_scores_per_image(raw_scores)
            dataset_entries.append({
                "image_id": f"{split}_{local_i:05d}",
                "split": split,
                "dataset_index": int(indices[local_i]),
                "label": int(labels_arr[local_i]),
                "norm_scores_per_class": norm_scores.tolist(),
                "raw_scores_per_class": raw_scores.tolist(),
            })

    batch_size = int(slurm_cfg["batch_size"])
    n_total = len(dataset_entries)
    n_batches = math.ceil(n_total / batch_size)
    batches: list[dict] = [
        {
            "batch_id": b,
            "image_ids": [
                dataset_entries[i]["image_id"]
                for i in range(b * batch_size, min((b + 1) * batch_size, n_total))
            ],
        }
        for b in range(n_batches)
    ]
    print(f"  total: {n_total}, batches: {n_batches} (batch_size={batch_size})")

    print("\n[4b/6] Score sanity guard ...")
    all_raw = np.array([e["raw_scores_per_class"] for e in dataset_entries], dtype=np.float64)
    all_norm = np.array([e["norm_scores_per_class"] for e in dataset_entries], dtype=np.float64)

    raw_finite = bool(np.isfinite(all_raw).all())
    raw_max_abs = float(np.abs(all_raw[np.isfinite(all_raw)]).max()) if raw_finite else float("inf")
    raw_std = float(all_raw[np.isfinite(all_raw)].std()) if raw_finite else 0.0
    n_unique_norm = int(len(np.unique(all_norm.round(8), axis=0)))

    score_diag = {
        "score_model_backend": score_model.get("backend", "unknown"),
        "score_train_accuracy": round(score_model["train_accuracy"], 4),
        "score_raw_max_abs": round(raw_max_abs, 4) if np.isfinite(raw_max_abs) else None,
        "score_raw_std": round(raw_std, 6),
        "score_norm_unique_rows": n_unique_norm,
        "score_guard_passed": False,
    }

    guard_errors: list[str] = []
    if not raw_finite:
        guard_errors.append("raw_scores contain non-finite values (NaN/Inf)")
    if raw_max_abs > 1e6:
        guard_errors.append(f"raw_scores max_abs={raw_max_abs:.2e} > 1e6 (likely overflow)")
    if raw_std < 1e-8:
        guard_errors.append(f"raw_scores std={raw_std:.2e} < 1e-8 (all identical)")
    if n_unique_norm <= 1:
        guard_errors.append(
            f"norm_scores_per_class has only {n_unique_norm} unique row(s) — "
            "all images would receive identical power patterns"
        )

    for line in [
        f"  backend          : {score_diag['score_model_backend']}",
        f"  train_accuracy   : {score_diag['score_train_accuracy']:.4f}",
        f"  raw_max_abs      : {raw_max_abs:.3e}",
        f"  raw_std          : {raw_std:.3e}",
        f"  norm unique rows : {n_unique_norm} / {n_total}",
    ]:
        print(line)

    if guard_errors:
        print("\n  SCORE GUARD FAILED:", file=sys.stderr)
        for err in guard_errors:
            print(f"    - {err}", file=sys.stderr)
        print(
            "\n  Likely cause: too few training images for stable Ridge fit.\n"
            "  Fix: increase --pilot-train-per-class to >= 20.\n"
            "  Aborting — GPU jobs will not start.",
            file=sys.stderr,
        )
        sys.exit(1)

    score_diag["score_guard_passed"] = True
    print("  Score guard      : PASSED")

    if args.dry_run:
        print("\n[DRY RUN] Would write:")
        print(f"  {run_dir}/manifest.json")
        print(f"  {run_dir}/dataset_index.json  ({n_total} entries)")
        print(f"  {run_dir}/batch_index.json    ({n_batches} batches)")
        print(f"  {run_dir}/projection_model.npz")
        print(f"  {run_dir}/feature_normalizer.npz")
        print(f"  {run_dir}/score_model.npz")
        if baseline_acc is not None:
            print(f"  Baseline (Ridge on PCA-{n_components}): {baseline_acc:.4f}")
        print(f"  Slurm tasks: {n_batches}")
        print(f"  Score guard: PASSED  (norm unique rows={n_unique_norm}/{n_total})")
        sys.exit(0)

    print("\n[5/6] Writing run directory ...")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "features"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    np.savez_compressed(
        os.path.join(run_dir, "projection_model.npz"),
        mean=pca_mean,
        components=pca_components,
        n_components=np.array([n_components]),
    )
    np.savez_compressed(
        os.path.join(run_dir, "feature_normalizer.npz"),
        f_min=f_min,
        f_max=f_max,
    )
    save_score_model(score_model, os.path.join(run_dir, "score_model.npz"))

    manifest: dict[str, Any] = {
        "config_path": args.config,
        "run_dir": run_dir,
        "pilot_mode": is_pilot,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_total": n_total,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "baseline_ridge_accuracy": baseline_acc,
        "ring_radius_um": ring_r,
        **score_diag,
        "prepare_complete": True,
        "gpu_complete": False,
        "finalize_complete": False,
    }
    atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)
    atomic_write_json(os.path.join(run_dir, "dataset_index.json"), dataset_entries)
    atomic_write_json(os.path.join(run_dir, "batch_index.json"), batches)

    print(f"  Manifest     : {run_dir}/manifest.json")
    print(f"  Dataset index: {n_total} entries")
    print(f"  Batches      : {n_batches}")
    if baseline_acc is not None:
        print(f"  Baseline     : {baseline_acc:.4f}")
    print("\n  prepare_run complete.")


if __name__ == "__main__":
    main()
