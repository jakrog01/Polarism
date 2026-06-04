"""CPU finalize: aggregate batch .npz, train PCA on far-field, fit classifier.

Invoked as:
    python -m mnist_ring_v1.stages.cpu.finalize --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL = True
except ImportError:
    _MPL = False


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _atomic_write_json(path: str, data: Any) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, cls=_NpEncoder)
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


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> list[list[int]]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def _numpy_pca(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    centered = X - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, len(s))
    components = Vt[:k]
    ev = (s[:k] ** 2) / max(len(X) - 1, 1)
    return mean, components, ev


def _nearest_centroid_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int = 10,
) -> np.ndarray:
    """Predict by assigning to the nearest class centroid (L2 distance)."""
    centroids = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for i, cls in enumerate(y_train):
        centroids[int(cls)] += X_train[i]
        counts[int(cls)] += 1
    for cls in range(n_classes):
        if counts[cls] > 0:
            centroids[cls] /= counts[cls]
    dists = np.sum((X_test[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return dists.argmin(axis=1)


def _centroid_distance_matrix(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 10,
) -> np.ndarray:
    """Return (n_classes, n_classes) pairwise L2 centroid distances."""
    centroids = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for i, cls in enumerate(y_train):
        centroids[int(cls)] += X_train[i]
        counts[int(cls)] += 1
    for cls in range(n_classes):
        if counts[cls] > 0:
            centroids[cls] /= counts[cls]
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def _per_digit_stats(scalars_rows: list[dict]) -> dict:
    per_digit: dict[int, dict] = {
        d: {
            "n": 0,
            "condensed": 0,
            "central_condensed": 0,
            "pretrigger_central": 0,
            "psi_sq_max": [],
            "psi_sq_central_max": [],
            "t_cond": [],
            "t_cond_central": [],
            "k_peak_um": [],
            "high_k_frac": [],
        }
        for d in range(10)
    }
    for s in scalars_rows:
        d = int(s.get("label", -1))
        if d < 0 or d > 9:
            continue
        per_digit[d]["n"] += 1
        if s.get("condensed"):
            per_digit[d]["condensed"] += 1
        if s.get("central_condensed"):
            per_digit[d]["central_condensed"] += 1
        if s.get("pretrigger_central_condensed"):
            per_digit[d]["pretrigger_central"] += 1
        per_digit[d]["psi_sq_max"].append(float(s.get("psi_sq_max", 0.0)))
        per_digit[d]["psi_sq_central_max"].append(float(s.get("psi_sq_central_max", 0.0)))
        if s.get("t_cond") is not None:
            per_digit[d]["t_cond"].append(float(s["t_cond"]))
        if s.get("t_cond_central") is not None:
            per_digit[d]["t_cond_central"].append(float(s["t_cond_central"]))
        per_digit[d]["k_peak_um"].append(float(s.get("k_peak_um", 0.0)))
        per_digit[d]["high_k_frac"].append(float(s.get("high_k_frac_0p8_nyq", 0.0)))

    result = {}
    for d, v in per_digit.items():
        n = v["n"]
        result[str(d)] = {
            "n": n,
            "condensed_fraction": v["condensed"] / max(n, 1),
            "central_condensed_fraction": v["central_condensed"] / max(n, 1),
            "pretrigger_central_fraction": v["pretrigger_central"] / max(n, 1),
            "psi_sq_max_mean": float(np.mean(v["psi_sq_max"])) if v["psi_sq_max"] else None,
            "psi_sq_max_median": float(np.median(v["psi_sq_max"])) if v["psi_sq_max"] else None,
            "psi_sq_central_max_mean": float(np.mean(v["psi_sq_central_max"])) if v["psi_sq_central_max"] else None,
            "t_cond_mean_ps": float(np.mean(v["t_cond"])) if v["t_cond"] else None,
            "t_cond_central_mean_ps": float(np.mean(v["t_cond_central"])) if v["t_cond_central"] else None,
            "k_peak_um_mean": float(np.mean(v["k_peak_um"])) if v["k_peak_um"] else None,
            "high_k_frac_0p8_mean": float(np.mean(v["high_k_frac"])) if v["high_k_frac"] else None,
        }
    return result


def _save_pca_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str = "Far-field PCA (2D) by digit",
) -> None:
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.colormaps.get_cmap("tab10").resampled(10)
    for d in range(10):
        mask = labels == d
        if mask.sum() == 0:
            continue
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=[cmap(d)], label=str(d), s=20, alpha=0.7
        )
    ax.legend(title="digit", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_centroid_distance_heatmap(
    dist_matrix: np.ndarray,
    out_path: str,
) -> None:
    if not _MPL:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist_matrix, cmap="viridis")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("digit")
    ax.set_ylabel("digit")
    ax.set_title("Centroid L2 distances (far-field PCA)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize mnist_ring_v1 run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pca-components", type=int, default=128)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" mnist_ring_v1 — finalize")
    print("=" * 60)

    feat_dir = os.path.join(run_dir, "features")
    batch_files = sorted(
        f for f in os.listdir(feat_dir) if f.startswith("batch_") and f.endswith(".npz")
    )
    if not batch_files:
        print("ERROR: no batch .npz files found", file=sys.stderr)
        sys.exit(1)
    print(f"  Aggregating {len(batch_files)} batch files ...")

    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_splits: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []

    for fname in batch_files:
        data = np.load(os.path.join(feat_dir, fname), allow_pickle=True)
        all_feats.append(data["kspace_features"])
        all_labels.append(data["labels"])
        all_splits.append(data["splits"])
        all_ids.append(data["image_ids"])

    X_all = np.concatenate(all_feats, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    splits_all = np.concatenate(all_splits, axis=0)
    ids_all = np.concatenate(all_ids, axis=0)

    train_mask = splits_all == "train"
    test_mask = splits_all == "test"
    print(f"  total: {len(X_all)}, train: {train_mask.sum()}, test: {test_mask.sum()}")

    meta_dir = os.path.join(run_dir, "metadata")
    scalars_rows: list[dict] = []
    power_provenances: list[dict] = []
    for fname in sorted(os.listdir(meta_dir)):
        if fname.startswith("batch_") and fname.endswith(".json"):
            with open(os.path.join(meta_dir, fname)) as f:
                bm = json.load(f)
            scalars_rows.extend(bm.get("scalars", []))
            if "power_provenance" in bm:
                power_provenances.append(bm["power_provenance"])

    power_provenance_summary = power_provenances[0] if power_provenances else {}
    n_condensed = sum(1 for s in scalars_rows if s.get("condensed"))
    n_central_condensed = sum(1 for s in scalars_rows if s.get("central_condensed"))
    n_pretrigger_central = sum(1 for s in scalars_rows if s.get("pretrigger_central_condensed"))
    psi_sq_vals = [s["psi_sq_max"] for s in scalars_rows]
    k_peak_vals = [s["k_peak_um"] for s in scalars_rows]
    high_k_vals = [s["high_k_frac_0p8_nyq"] for s in scalars_rows]
    t_cond_vals = [s["t_cond"] for s in scalars_rows if s.get("t_cond") is not None]
    t_cond_central_vals = [s["t_cond_central"] for s in scalars_rows if s.get("t_cond_central") is not None]
    has_central_metrics = any("central_condensed" in s for s in scalars_rows)

    n_pca = min(args.pca_components, train_mask.sum(), X_all.shape[1])
    print(f"\n[1/4] PCA on far-field (train only, {n_pca} components) ...")
    mean_ff, comps_ff, ev_ff = _numpy_pca(X_all[train_mask], n_pca)
    X_tr_ff = (X_all[train_mask] - mean_ff) @ comps_ff.T
    X_te_ff = (X_all[test_mask] - mean_ff) @ comps_ff.T
    X_all_ff = (X_all - mean_ff) @ comps_ff.T

    print("[2/4] Scalar features + centroid analysis ...")
    id_to_scalars = {s["image_id"]: s for s in scalars_rows}

    def _scalar_row(img_id: str) -> np.ndarray:
        s = id_to_scalars.get(str(img_id), {})
        return np.array(
            [
                s.get("psi_sq_max", 0.0),
                s.get("k_peak_um", 0.0),
                s.get("k_centroid_um", 0.0),
                s.get("high_k_frac_0p8_nyq", 0.0),
            ],
            dtype=np.float64,
        )

    scalar_tr = np.stack([_scalar_row(i) for i in ids_all[train_mask]], axis=0)
    X_tr = np.concatenate([X_tr_ff, scalar_tr], axis=1)
    y_tr = y_all[train_mask]

    has_test = test_mask.sum() > 0

    if has_test:
        scalar_te = np.stack([_scalar_row(i) for i in ids_all[test_mask]], axis=0)
        X_te = np.concatenate([X_te_ff, scalar_te], axis=1)
        y_te = y_all[test_mask]
    else:
        X_te = np.zeros((0, X_tr.shape[1]), dtype=np.float64)
        y_te = np.zeros(0, dtype=np.int64)

    nc_pred_tr = _nearest_centroid_predict(X_tr, y_tr, X_tr)
    nc_acc_train = float(np.mean(nc_pred_tr == y_tr))

    if has_test:
        nc_pred_te = _nearest_centroid_predict(X_tr, y_tr, X_te)
        nc_acc_test = float(np.mean(nc_pred_te == y_te))
        nc_cm = _confusion_matrix(y_te, nc_pred_te)
    else:
        nc_acc_test = None
        nc_cm = None

    dist_matrix = _centroid_distance_matrix(X_tr, y_tr)
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix_list = dist_matrix.tolist()
    mean_inter_centroid_dist = (
        float(dist_matrix[dist_matrix > 0].mean()) if (dist_matrix > 0).any() else 0.0
    )

    print(f"[3/4] Ridge classifier ({X_tr.shape[1]} features) ...")
    if _SKLEARN:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_tr_s, y_tr)
        y_pred_tr = clf.predict(X_tr_s)
        if has_test:
            X_te_s = scaler.transform(X_te)
            y_pred_te = clf.predict(X_te_s)
        else:
            y_pred_te = np.zeros(0, dtype=np.int64)
    else:
        from scipy.linalg import lstsq

        n_cls = 10
        Y_oh = np.eye(n_cls)[y_tr]
        X_tr_b = np.hstack([X_tr, np.ones((len(X_tr), 1))])
        W, _, _, _ = lstsq(X_tr_b, Y_oh)
        y_pred_tr = (X_tr_b @ W).argmax(axis=1)
        if has_test:
            X_te_b = np.hstack([X_te, np.ones((len(X_te), 1))])
            y_pred_te = (X_te_b @ W).argmax(axis=1)
        else:
            y_pred_te = np.zeros(0, dtype=np.int64)

    acc_train = float(np.mean(y_pred_tr == y_tr))
    acc_test = float(np.mean(y_pred_te == y_te)) if has_test else None
    ridge_cm = _confusion_matrix(y_te, y_pred_te) if has_test else None

    print("[4/4] Diagnostics and plots ...")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    X_all_2d = X_all_ff[:, :2]
    scatter_path = os.path.join(plots_dir, "pca_scatter_train.png")
    _save_pca_scatter(
        X_all_2d[train_mask],
        y_all[train_mask],
        scatter_path,
        "Far-field PCA PC1/PC2 — train set",
    )
    centroid_path = os.path.join(plots_dir, "centroid_distances.png")
    _save_centroid_distance_heatmap(dist_matrix, centroid_path)

    per_digit = _per_digit_stats(scalars_rows)

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    baseline_acc = manifest.get("baseline_pca_accuracy")

    summary = {
        "nearest_centroid": {
            "accuracy_train": nc_acc_train,
            "accuracy_test": nc_acc_test,
            "confusion_matrix": nc_cm,
            "mean_inter_centroid_dist": mean_inter_centroid_dist,
            "centroid_distance_matrix": dist_matrix_list,
            "atlas_only": not has_test,
        },
        "ridge_classifier": {
            "accuracy_train": acc_train,
            "accuracy_test": acc_test,
            "confusion_matrix": ridge_cm,
            "atlas_only": not has_test,
        },
        "baseline_pca_accuracy": baseline_acc,
        "pilot_note": (
            "Small pilot run — metrics are response atlas indicators, not final benchmark accuracy."
            if manifest.get("pilot_mode") else None
        ),
        "n_pca_farfield_components": n_pca,
        "n_total": len(X_all),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_condensed": n_condensed,
        "fraction_condensed": n_condensed / max(len(scalars_rows), 1),
        "n_central_condensed": n_central_condensed,
        "central_condensed_fraction": n_central_condensed / max(len(scalars_rows), 1),
        "n_pretrigger_central": n_pretrigger_central,
        "pretrigger_central_fraction": n_pretrigger_central / max(len(scalars_rows), 1),
        "has_central_metrics": has_central_metrics,
        "per_digit_stats": per_digit,
        "global_scalars": {
            "psi_sq_max_mean": float(np.mean(psi_sq_vals)) if psi_sq_vals else None,
            "psi_sq_max_median": float(np.median(psi_sq_vals)) if psi_sq_vals else None,
            "psi_sq_max_p10": float(np.percentile(psi_sq_vals, 10)) if psi_sq_vals else None,
            "psi_sq_max_p90": float(np.percentile(psi_sq_vals, 90)) if psi_sq_vals else None,
            "k_peak_um_mean": float(np.mean(k_peak_vals)) if k_peak_vals else None,
            "high_k_frac_0p8_nyq_mean": float(np.mean(high_k_vals)) if high_k_vals else None,
            "high_k_frac_0p8_nyq_max": float(np.max(high_k_vals)) if high_k_vals else None,
            "t_cond_mean_ps": float(np.mean(t_cond_vals)) if t_cond_vals else None,
            "t_cond_std_ps": float(np.std(t_cond_vals)) if len(t_cond_vals) > 1 else None,
            "t_cond_central_mean_ps": float(np.mean(t_cond_central_vals)) if t_cond_central_vals else None,
            "t_cond_central_std_ps": float(np.std(t_cond_central_vals)) if len(t_cond_central_vals) > 1 else None,
        },
        "power_provenance": power_provenance_summary,
        "plots": {
            "pca_scatter_train": "plots/pca_scatter_train.png" if _MPL else None,
            "centroid_distances": "plots/centroid_distances.png" if _MPL else None,
            "pump_geometry": "plots/pump_geometry.png" if _MPL else None,
        },
    }

    out_path = os.path.join(run_dir, "results_summary.json")
    _atomic_write_json(out_path, summary)

    print(f"\n  Nearest-centroid  train: {nc_acc_train:.4f}", end="")
    if has_test:
        print(f"  test: {nc_acc_test:.4f}")
    else:
        print("  test: N/A (atlas-only, no test split)")
    print(f"  Ridge classifier  train: {acc_train:.4f}", end="")
    if has_test:
        print(f"  test: {acc_test:.4f}")
    else:
        print("  test: N/A")
    if baseline_acc is not None and has_test and nc_acc_test is not None:
        delta = nc_acc_test - baseline_acc
        print(f"  Baseline (PCA-16) test : {baseline_acc:.4f}  (NC delta={delta:+.4f})")
    print(f"  Mean inter-centroid dist : {mean_inter_centroid_dist:.4f}")
    print(
        f"  Condensed  : {n_condensed}/{len(scalars_rows)} ({100*n_condensed/max(len(scalars_rows),1):.1f}%)"
    )

    if high_k_vals:
        hk_max = float(np.max(high_k_vals))
        if hk_max >= 1e-3:
            print(
                f"  WARNING: high_k_frac_0p8_nyq max={hk_max:.2e} >= 1e-3 — check solver/geometry"
            )

    per_digit_cond = {d: v["condensed_fraction"] for d, v in per_digit.items()}
    worst_digit = min(per_digit_cond, key=per_digit_cond.get)
    print(
        f"  Per-digit condensed fraction: min={per_digit_cond[worst_digit]:.2f} (digit {worst_digit})"
    )

    if has_central_metrics:
        n_total_imgs = max(len(scalars_rows), 1)
        print(
            f"  Central condensed    : {n_central_condensed}/{n_total_imgs} "
            f"({100 * n_central_condensed / n_total_imgs:.1f}%)"
        )
        print(
            f"  Pre-trigger central  : {n_pretrigger_central}/{n_total_imgs} "
            f"({100 * n_pretrigger_central / n_total_imgs:.1f}%)"
        )
        if n_pretrigger_central > 0:
            print(
                f"  WARNING: {n_pretrigger_central} images had central condensation BEFORE trigger — "
                "ring power may be too high or threshold calibration needs re-run."
            )
    else:
        print("  Central metrics: not available (old run or batch ran without central detector)")

    if manifest.get("pilot_mode"):
        print(
            "\n  NOTE: pilot run — these metrics are a response atlas, "
            "not a final classification benchmark."
        )

    print(f"\n  Summary: {out_path}")
    if _MPL:
        print(f"  Plots  : {plots_dir}/")
    else:
        print("  Plots  : matplotlib not available — install to generate PCA scatter")


if __name__ == "__main__":
    main()
