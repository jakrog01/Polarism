"""CPU finalize: aggregate WTA batch results and compute accuracy metrics.

Invoked as:
    python -m mnist_wta_v1.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

from mnist_common.io.atomic import atomic_write_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < n_classes and 0 <= int(p) < n_classes:
            cm[int(t), int(p)] += 1
    return cm.tolist()


def _per_class_stats(scalars: list[dict]) -> dict:
    stats: dict[int, dict] = {d: {"n": 0, "correct": 0, "no_cond": 0, "multi_cond": 0, "margins": []} for d in range(10)}
    for s in scalars:
        d = int(s.get("label", -1))
        if d < 0 or d > 9:
            continue
        stats[d]["n"] += 1
        if s.get("correct"):
            stats[d]["correct"] += 1
        if s.get("no_cond"):
            stats[d]["no_cond"] += 1
        if s.get("multi_cond"):
            stats[d]["multi_cond"] += 1
        m = s.get("margin")
        if m is not None:
            stats[d]["margins"].append(float(m))

    result = {}
    for d, v in stats.items():
        n = v["n"]
        margins = v["margins"]
        result[str(d)] = {
            "n": n,
            "accuracy": v["correct"] / max(n, 1),
            "no_cond_rate": v["no_cond"] / max(n, 1),
            "multi_cond_rate": v["multi_cond"] / max(n, 1),
            "margin_mean": float(np.mean(margins)) if margins else None,
            "margin_median": float(np.median(margins)) if margins else None,
        }
    return result


def _save_confusion_plot(cm: list[list[int]], out_path: str, title: str) -> None:
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize mnist_wta_v1 run")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" mnist_wta_v1 — finalize")
    print("=" * 60)

    feat_dir = os.path.join(run_dir, "features")
    batch_files = sorted(f for f in os.listdir(feat_dir) if f.startswith("batch_") and f.endswith(".npz"))
    if not batch_files:
        print("ERROR: no batch .npz files found", file=sys.stderr)
        sys.exit(1)
    print(f"  Aggregating {len(batch_files)} batch files ...")

    all_roi: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_winners: list[np.ndarray] = []
    all_splits: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []

    for fname in batch_files:
        data = np.load(os.path.join(feat_dir, fname), allow_pickle=True)
        all_roi.append(data["roi_features"])
        all_labels.append(data["labels"])
        all_winners.append(data["winners"])
        all_splits.append(data["splits"])
        all_ids.append(data["image_ids"])

    roi_all = np.concatenate(all_roi, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)
    winners_all = np.concatenate(all_winners, axis=0)
    splits_all = np.concatenate(all_splits, axis=0)

    train_mask = splits_all == "train"
    test_mask = splits_all == "test"
    print(f"  total: {len(roi_all)}  train: {train_mask.sum()}  test: {test_mask.sum()}")

    meta_dir = os.path.join(run_dir, "metadata")
    scalars_all: list[dict] = []
    for fname in sorted(os.listdir(meta_dir)):
        if fname.startswith("batch_") and fname.endswith(".json"):
            with open(os.path.join(meta_dir, fname)) as f:
                bm = json.load(f)
            scalars_all.extend(bm.get("scalars", []))

    has_test = test_mask.sum() > 0
    correct_train = int(np.sum(winners_all[train_mask] == labels_all[train_mask]))
    acc_train = correct_train / max(int(train_mask.sum()), 1)
    acc_test = None
    cm_test = None
    if has_test:
        correct_test = int(np.sum(winners_all[test_mask] == labels_all[test_mask]))
        acc_test = correct_test / max(int(test_mask.sum()), 1)
        cm_test = _confusion_matrix(labels_all[test_mask], winners_all[test_mask])

    test_scalars = [s for s in scalars_all if s.get("split") == "test"]
    n_no_cond = sum(1 for s in test_scalars if s.get("no_cond"))
    n_multi_cond = sum(1 for s in test_scalars if s.get("multi_cond"))
    margins_test = [float(s["margin"]) for s in test_scalars if s.get("margin") is not None]

    def _conditional_accuracy(scalars: list[dict], condition_fn) -> float | None:
        subset = [s for s in scalars if condition_fn(s)]
        if not subset:
            return None
        return float(np.mean([int(s.get("correct", 0)) for s in subset]))

    condensed = lambda s: not s.get("no_cond", True)
    margin_gt_2 = lambda s: float(s.get("margin", 0)) > 2.0
    margin_gt_5 = lambda s: float(s.get("margin", 0)) > 5.0

    acc_test_excl_no_cond      = _conditional_accuracy(test_scalars, condensed)
    acc_test_margin_gt_2       = _conditional_accuracy(test_scalars, margin_gt_2)
    acc_test_margin_gt_5       = _conditional_accuracy(test_scalars, margin_gt_5)
    acc_test_margin_gt_2_excl  = _conditional_accuracy(test_scalars, lambda s: condensed(s) and margin_gt_2(s))
    acc_test_margin_gt_5_excl  = _conditional_accuracy(test_scalars, lambda s: condensed(s) and margin_gt_5(s))

    n_excl_no_cond   = sum(1 for s in test_scalars if condensed(s))
    n_margin_gt_2    = sum(1 for s in test_scalars if margin_gt_2(s))
    n_margin_gt_5    = sum(1 for s in test_scalars if margin_gt_5(s))
    n_m2_excl        = sum(1 for s in test_scalars if condensed(s) and margin_gt_2(s))
    n_m5_excl        = sum(1 for s in test_scalars if condensed(s) and margin_gt_5(s))

    per_class = _per_class_stats([s for s in scalars_all if s.get("split") == "test"])

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    baseline_acc = manifest.get("baseline_ridge_accuracy")

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    if cm_test is not None:
        _save_confusion_plot(cm_test, os.path.join(plots_dir, "confusion_matrix.png"), "WTA confusion matrix (test)")

    summary: dict[str, Any] = {
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
        "baseline_ridge_accuracy": baseline_acc,
        "delta_vs_baseline": (acc_test - baseline_acc) if (acc_test is not None and baseline_acc is not None) else None,
        "accuracy_test_excl_no_cond": acc_test_excl_no_cond,
        "accuracy_test_margin_gt_2": acc_test_margin_gt_2,
        "accuracy_test_margin_gt_5": acc_test_margin_gt_5,
        "accuracy_test_margin_gt_2_excl_no_cond": acc_test_margin_gt_2_excl,
        "accuracy_test_margin_gt_5_excl_no_cond": acc_test_margin_gt_5_excl,
        "n_total": len(roi_all),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_test_excl_no_cond": n_excl_no_cond,
        "n_test_margin_gt_2": n_margin_gt_2,
        "n_test_margin_gt_5": n_margin_gt_5,
        "n_test_margin_gt_2_excl_no_cond": n_m2_excl,
        "n_test_margin_gt_5_excl_no_cond": n_m5_excl,
        "no_cond_rate_test": n_no_cond / max(len(test_scalars), 1),
        "multi_cond_rate_test": n_multi_cond / max(len(test_scalars), 1),
        "margin_mean_test": float(np.mean(margins_test)) if margins_test else None,
        "margin_median_test": float(np.median(margins_test)) if margins_test else None,
        "confusion_matrix_test": cm_test,
        "per_class_stats": per_class,
    }

    atomic_write_json(os.path.join(run_dir, "results_summary_wta.json"), summary)

    print(f"\n  Accuracy  train: {acc_train:.4f}")
    if has_test:
        print(f"  Accuracy  test : {acc_test:.4f}")
        if baseline_acc is not None:
            print(f"  Baseline  test : {baseline_acc:.4f}  (delta={acc_test - baseline_acc:+.4f})")
        if acc_test_excl_no_cond is not None:
            print(f"  Acc (excl no_cond, n={n_excl_no_cond}): {acc_test_excl_no_cond:.4f}")
        if acc_test_margin_gt_2 is not None:
            print(f"  Acc (margin>2,           n={n_margin_gt_2:4d}): {acc_test_margin_gt_2:.4f}")
        if acc_test_margin_gt_2_excl is not None:
            print(f"  Acc (margin>2 ∩ condensed n={n_m2_excl:4d}): {acc_test_margin_gt_2_excl:.4f}")
        if acc_test_margin_gt_5 is not None:
            print(f"  Acc (margin>5,           n={n_margin_gt_5:4d}): {acc_test_margin_gt_5:.4f}")
        if acc_test_margin_gt_5_excl is not None:
            print(f"  Acc (margin>5 ∩ condensed n={n_m5_excl:4d}): {acc_test_margin_gt_5_excl:.4f}")
    print(f"  No-cond rate (test): {summary['no_cond_rate_test']:.3f}")
    print(f"  Multi-cond rate    : {summary['multi_cond_rate_test']:.3f}")
    if margins_test:
        print(f"  Margin mean/median : {summary['margin_mean_test']:.2f} / {summary['margin_median_test']:.2f}")
    print(f"\n  Summary: {run_dir}/results_summary_wta.json")


if __name__ == "__main__":
    main()
