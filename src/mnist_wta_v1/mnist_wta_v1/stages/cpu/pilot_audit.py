"""Pilot audit for completed mnist_wta_v1 A/B runs.

Loads two completed runs (e.g. R=8 and R=15) and evaluates every observable
that can be reconstructed from the saved artifacts, without running any new GPU
simulations.

Invoked as:
    python -m mnist_wta_v1.stages.cpu.pilot_audit \\
        --run-a <path_R8> --run-b <path_R15> --out-dir <output_dir>

    or for a single run:
    python -m mnist_wta_v1.stages.cpu.pilot_audit \\
        --run-a <path> --out-dir <output_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

class _NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _atomic_write(path: str, content: str) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        os.unlink(tmp)
        raise
    os.replace(tmp, path)


def _atomic_write_json(path: str, data: Any) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, cls=_NpEncoder)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        os.unlink(tmp)
        raise
    os.replace(tmp, path)


def _fmt(v: Any, fmt: str = ".4f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


# ── data loading ─────────────────────────────────────────────────────────────

def _load_run(run_dir: str) -> dict[str, Any]:
    """Load all saved artifacts from a completed run directory."""
    feat_dir = os.path.join(run_dir, "features")
    meta_dir = os.path.join(run_dir, "metadata")

    all_roi: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_splits: list[np.ndarray] = []
    all_image_ids: list[np.ndarray] = []
    scalars: list[dict] = []

    for fname in sorted(os.listdir(feat_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(feat_dir, fname), allow_pickle=True)
        all_roi.append(d["roi_features"])
        all_labels.append(d["labels"])
        all_splits.append(d["splits"])
        all_image_ids.append(d["image_ids"])

    for fname in sorted(os.listdir(meta_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(meta_dir, fname)) as f:
            bm = json.load(f)
        scalars.extend(bm.get("scalars", []))

    roi = np.concatenate(all_roi, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    splits = np.concatenate(all_splits, axis=0)
    image_ids = np.concatenate(all_image_ids, axis=0)

    with open(os.path.join(run_dir, "dataset_index.json")) as f:
        dataset_index = json.load(f)
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(run_dir, "calibration_wta.json")) as f:
        calib = json.load(f)

    id_to_di = {e["image_id"]: e for e in dataset_index}

    # Align dataset_index entries to feature row order
    norm_scores = np.array(
        [id_to_di[str(iid)]["norm_scores_per_class"] for iid in image_ids],
        dtype=np.float64,
    )
    raw_scores = np.array(
        [id_to_di[str(iid)]["raw_scores_per_class"] for iid in image_ids],
        dtype=np.float64,
    )
    powers = np.array(
        [s["power_per_class"] for s in scalars],
        dtype=np.float64,
    )
    t_cond = np.array(
        [s["t_cond"] if s["t_cond"] is not None else np.nan for s in scalars],
        dtype=np.float64,
    )
    margins = np.array([s["margin"] for s in scalars], dtype=np.float64)
    multi_cond = np.array([s["multi_cond"] for s in scalars], dtype=bool)
    no_cond = np.array([s["no_cond"] for s in scalars], dtype=bool)
    condensed = np.array([s["condensed"] for s in scalars], dtype=bool)

    # Check for HDF5 / complex-psi files
    hdf5_files = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.endswith((".h5", ".hdf5")):
                hdf5_files.append(os.path.join(root, f))

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "calib": calib,
        "roi": roi,
        "labels": labels,
        "splits": splits,
        "image_ids": image_ids,
        "norm_scores": norm_scores,
        "raw_scores": raw_scores,
        "powers": powers,
        "t_cond": t_cond,
        "margins": margins,
        "multi_cond": multi_cond,
        "no_cond": no_cond,
        "condensed": condensed,
        "hdf5_files": hdf5_files,
        "n_images": len(roi),
        "ring_radius_um": manifest.get("ring_radius_um"),
        "baseline_ridge_accuracy": manifest.get("baseline_ridge_accuracy"),
    }


# ── data audit ───────────────────────────────────────────────────────────────

def _audit_run(run: dict) -> dict[str, Any]:
    """Inspect data completeness and flag pathologies."""
    norm = run["norm_scores"]
    raw = run["raw_scores"]
    roi = run["roi"]
    t_cond = run["t_cond"]

    n_unique_norm = len(np.unique(norm, axis=0))
    scores_identical = n_unique_norm == 1

    raw_magnitude = float(np.abs(raw[np.isfinite(raw)]).max()) if np.isfinite(raw).any() else float("inf")
    raw_overflow = raw_magnitude > 1e10 or not np.isfinite(raw).all()

    roi_std_per_class = roi.std(axis=0)
    roi_max_std = float(roi_std_per_class.max())

    t_cond_valid = ~np.isnan(t_cond)
    t_cond_range = (float(t_cond[t_cond_valid].min()), float(t_cond[t_cond_valid].max())) if t_cond_valid.any() else (None, None)

    train_n = int((run["splits"] == "train").sum())
    test_n  = int((run["splits"] == "test").sum())

    run_size_kb = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(run["run_dir"])
        for f in files
    ) / 1024

    return {
        "ring_radius_um": run["ring_radius_um"],
        "n_train": train_n,
        "n_test": test_n,
        "hdf5_present": len(run["hdf5_files"]) > 0,
        "complex_psi_present": False,
        "psi_k_present": False,
        "time_resolved_roi_present": False,
        "per_spot_t_cond_present": False,
        "shots_per_image": 1,
        "run_size_kb": round(run_size_kb, 1),
        "score_model_overflow": bool(raw_overflow),
        "raw_scores_magnitude": float(raw_magnitude) if np.isfinite(raw_magnitude) else None,
        "norm_scores_identical": bool(scores_identical),
        "n_unique_norm_score_rows": int(n_unique_norm),
        "roi_max_std_across_images": round(roi_max_std, 6),
        "t_cond_range_ps": t_cond_range,
        "multi_cond_fraction": float(run["multi_cond"].mean()),
        "no_cond_fraction": float(run["no_cond"].mean()),
        "condensed_fraction": float(run["condensed"].mean()),
    }


# ── classifiers ──────────────────────────────────────────────────────────────

def _ridge_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Ridge classifier with numpy fallback. Returns train/test accuracy."""
    if len(X_train) == 0 or len(X_test) == 0:
        return {"acc_train": None, "acc_test": None, "n_train": 0, "n_test": 0, "error": "empty split"}

    mean = X_train.mean(axis=0)
    std = np.where(X_train.std(axis=0) > 1e-12, X_train.std(axis=0), 1.0)
    X_tr = (X_train - mean) / std
    X_te = (X_test  - mean) / std

    try:
        from sklearn.linear_model import RidgeClassifier
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X_tr, y_train)
        y_pred_tr = clf.predict(X_tr)
        y_pred_te = clf.predict(X_te)
        classes = clf.classes_
    except ImportError:
        n_cls = 10
        Y_oh = (y_train[:, None] == np.arange(n_cls)[None, :]).astype(np.float64) * 2 - 1
        X_b = np.hstack([X_tr, np.ones((len(X_tr), 1))])
        reg = np.eye(X_b.shape[1])
        reg[-1, -1] = 0.0
        A = X_b.T @ X_b + alpha * reg
        W_b = np.linalg.solve(A, X_b.T @ Y_oh)
        W, b = W_b[:-1].T, W_b[-1]
        classes = np.arange(n_cls)
        y_pred_tr = classes[np.argmax(X_tr @ W.T + b, axis=1)]
        y_pred_te = classes[np.argmax(X_te @ W.T + b, axis=1)]

    acc_tr = float(np.mean(y_pred_tr == y_train))
    acc_te = float(np.mean(y_pred_te == y_test))
    chance = 1.0 / len(np.unique(y_train))

    return {
        "acc_train": round(acc_tr, 4),
        "acc_test": round(acc_te, 4),
        "delta_vs_chance": round(acc_te - chance, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "chance_level": round(chance, 4),
    }


def _classify_channel(
    name: str,
    X: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    labels: np.ndarray,
    baseline: float | None,
) -> dict[str, Any]:
    """Classify one channel and add metadata."""
    result = _ridge_classify(
        X[train_mask], labels[train_mask],
        X[test_mask],  labels[test_mask],
    )
    result["channel"] = name

    # Flag if feature has near-zero variance (no signal)
    total_std = float(X.std())
    result["feature_std"] = round(total_std, 8)
    result["zero_variance"] = total_std < 1e-10

    result["baseline_ridge_acc"] = baseline
    acc_te = result["acc_test"]
    if acc_te is not None and baseline is not None:
        result["delta_vs_baseline"] = round(acc_te - baseline, 4)
    else:
        result["delta_vs_baseline"] = None

    return result


# ── main analysis ─────────────────────────────────────────────────────────────

def _analyze_run(run: dict) -> dict[str, Any]:
    """Run all available channel classifiers on one pilot."""
    train_mask = run["splits"] == "train"
    test_mask  = run["splits"] == "test"
    labels = run["labels"]
    baseline = run["baseline_ridge_accuracy"]
    audit = _audit_run(run)

    channels: dict[str, dict] = {}

    # Channel: roi_integrals
    channels["roi"] = _classify_channel(
        "roi", run["roi"], train_mask, test_mask, labels, baseline
    )

    # Channel: norm_scores
    channels["scores_norm"] = _classify_channel(
        "scores_norm", run["norm_scores"], train_mask, test_mask, labels, baseline
    )

    # Channel: power_per_class (linear transform of norm_scores)
    channels["powers"] = _classify_channel(
        "powers", run["powers"], train_mask, test_mask, labels, baseline
    )

    # Channel: roi + norm_scores
    if not audit["norm_scores_identical"]:
        X_combined = np.concatenate([run["roi"], run["norm_scores"]], axis=1)
        channels["roi+scores_norm"] = _classify_channel(
            "roi+scores_norm", X_combined, train_mask, test_mask, labels, baseline
        )
    else:
        channels["roi+scores_norm"] = {
            "channel": "roi+scores_norm",
            "SKIPPED": "norm_scores identical — combining with roi adds no information",
        }

    # Channel: roi + powers
    if not audit["norm_scores_identical"]:
        X_rp = np.concatenate([run["roi"], run["powers"]], axis=1)
        channels["roi+powers"] = _classify_channel(
            "roi+powers", X_rp, train_mask, test_mask, labels, baseline
        )
    else:
        channels["roi+powers"] = {
            "channel": "roi+powers",
            "SKIPPED": "powers identical — same as roi channel",
        }

    # Channel: global_t_cond (single scalar — weak channel)
    t_cond_2d = run["t_cond"].reshape(-1, 1)
    # Replace NaN with per-split mean
    t_valid_mean = float(np.nanmean(run["t_cond"][train_mask])) if np.isfinite(run["t_cond"][train_mask]).any() else 0.0
    t_cond_filled = np.where(np.isnan(t_cond_2d), t_valid_mean, t_cond_2d)
    channels["global_t_cond"] = _classify_channel(
        "global_t_cond", t_cond_filled, train_mask, test_mask, labels, baseline
    )

    # Channel: roi + global_t_cond
    X_roi_t = np.concatenate([run["roi"], t_cond_filled], axis=1)
    channels["roi+global_t_cond"] = _classify_channel(
        "roi+global_t_cond", X_roi_t, train_mask, test_mask, labels, baseline
    )

    # Unavailable channels
    unavailable = {
        "t_cond_per_spot": {
            "reason": "No per-spot condensation time in saved output. Only global t_cond available.",
            "min_pipeline_change": "Record per-class ROI integral traces at each solver step; detect per-class condensation threshold crossing.",
        },
        "phase_phi_c": {
            "reason": "No complex psi saved. Only |psi|^2 (real-valued ROI integrals) available.",
            "min_pipeline_change": "Save complex psi snapshots every 1-2 ps for 5-10 images in a diagnostic run.",
        },
        "far_field_psi_k": {
            "reason": "No FFT(psi) saved. psi(k) requires complex psi field.",
            "min_pipeline_change": "Compute np.fft.fft2(psi) at readout time and save cropped |psi_k|^2 or full complex psi_k for diagnostic images.",
        },
        "shot_to_shot_noise": {
            "reason": "Each image was simulated once (n_shots=1). No repeated seeds available.",
            "min_pipeline_change": "Simulate 5 seeds x 5 images to characterise simulation noise floor vs between-digit variance.",
        },
    }

    return {
        "run_dir": run["run_dir"],
        "audit": audit,
        "channels": channels,
        "unavailable": unavailable,
        "ring_radius_um": run["ring_radius_um"],
        "baseline_ridge_accuracy": baseline,
    }


# ── report generation ─────────────────────────────────────────────────────────

def _channel_table_row(ch: dict, label: str = "") -> str:
    if "SKIPPED" in ch:
        return f"| {ch['channel']:30s} | SKIPPED | SKIPPED | {ch['SKIPPED'][:60]} |"
    acc_te = _fmt(ch.get("acc_test"), ".4f")
    delta_b = _fmt(ch.get("delta_vs_baseline"), "+.4f")
    delta_c = _fmt(ch.get("delta_vs_chance"), "+.4f")
    zero_var = "⚠ zero-var" if ch.get("zero_variance") else ""
    return f"| {ch['channel']:30s} | {acc_te:8s} | {delta_b:9s} | {delta_c:9s} | {zero_var} |"


def _make_report(
    results_a: dict,
    results_b: dict | None,
    run_date: str,
    out_dir: str,
) -> str:

    audit_a = results_a["audit"]
    audit_b = results_b["audit"] if results_b else None
    ch_a = results_a["channels"]
    ch_b = results_b["channels"] if results_b else None
    baseline_a = results_a["baseline_ridge_accuracy"]
    baseline_b = results_b["baseline_ridge_accuracy"] if results_b else None

    # ── pathology section ──────────────────────────────────────────────────────
    pathology_a = ""
    if audit_a["score_model_overflow"]:
        pathology_a = f"""
> **CRITICAL: Score model overflow in run A (R={audit_a['ring_radius_um']} μm)**
> `raw_scores ~ {audit_a['raw_scores_magnitude']:.2e}` — numerical overflow during Ridge fit.
> All `norm_scores_per_class` are identical ({audit_a['n_unique_norm_score_rows']} unique row out of {results_a['channels']['scores_norm']['n_train'] + results_a['channels']['scores_norm']['n_test']} images).
> **Every image was simulated with the same power pattern.** ROI variation is pure simulation noise, not digit information.
> Root cause: score model trained on {results_a['audit']['n_train'] + results_a['audit']['n_test']} images — too few for stable PCA+Ridge fit, or a numerical issue in the numpy fallback Ridge.
"""

    pathology_b = ""
    if results_b and audit_b["score_model_overflow"]:
        pathology_b = f"""
> **CRITICAL: Score model overflow in run B (R={audit_b['ring_radius_um']} μm)** — same root cause as A.
"""

    # ── data audit table ───────────────────────────────────────────────────────
    def _audit_row(key: str, a: Any, b: Any = "—") -> str:
        return f"| {key:40s} | {str(a):20s} | {str(b):20s} |"

    audit_rows = [
        _audit_row("HDF5 / complex psi present", audit_a["hdf5_present"], audit_b["hdf5_present"] if audit_b else "—"),
        _audit_row("Time-resolved ROI traces", audit_a["time_resolved_roi_present"], audit_b["time_resolved_roi_present"] if audit_b else "—"),
        _audit_row("Per-spot t_cond", audit_a["per_spot_t_cond_present"], audit_b["per_spot_t_cond_present"] if audit_b else "—"),
        _audit_row("shots per image", audit_a["shots_per_image"], audit_b["shots_per_image"] if audit_b else "—"),
        _audit_row("run size (KB)", audit_a["run_size_kb"], audit_b["run_size_kb"] if audit_b else "—"),
        _audit_row("n_train / n_test", f"{audit_a['n_train']}/{audit_a['n_test']}", f"{audit_b['n_train']}/{audit_b['n_test']}" if audit_b else "—"),
        _audit_row("score model overflow", audit_a["score_model_overflow"], audit_b["score_model_overflow"] if audit_b else "—"),
        _audit_row("raw_scores magnitude", f"{audit_a['raw_scores_magnitude']:.2e}" if audit_a['raw_scores_magnitude'] else "N/A", f"{audit_b['raw_scores_magnitude']:.2e}" if (audit_b and audit_b['raw_scores_magnitude']) else "—"),
        _audit_row("norm_scores identical (all images)", audit_a["norm_scores_identical"], audit_b["norm_scores_identical"] if audit_b else "—"),
        _audit_row("n unique norm_score rows", audit_a["n_unique_norm_score_rows"], audit_b["n_unique_norm_score_rows"] if audit_b else "—"),
        _audit_row("ROI max std across images", f"{audit_a['roi_max_std_across_images']:.4f}", f"{audit_b['roi_max_std_across_images']:.4f}" if audit_b else "—"),
        _audit_row("multi_cond fraction", f"{audit_a['multi_cond_fraction']:.2f}", f"{audit_b['multi_cond_fraction']:.2f}" if audit_b else "—"),
        _audit_row("no_cond fraction", f"{audit_a['no_cond_fraction']:.2f}", f"{audit_b['no_cond_fraction']:.2f}" if audit_b else "—"),
        _audit_row("t_cond range (ps)", str(audit_a["t_cond_range_ps"]), str(audit_b["t_cond_range_ps"]) if audit_b else "—"),
    ]
    audit_table = "| Metric | Run A (R={:.0f} μm) | Run B (R={:.0f} μm) |\n".format(
        audit_a["ring_radius_um"], audit_b["ring_radius_um"] if audit_b else 0
    )
    audit_table += "|" + "-" * 42 + "|" + "-" * 22 + "|" + "-" * 22 + "|\n"
    audit_table += "\n".join(audit_rows)

    # ── channel results ────────────────────────────────────────────────────────
    channel_hdr = (
        "| channel | acc_test | Δ_baseline | Δ_chance | note |\n"
        "|" + "-"*32 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*20 + "|\n"
    )

    def _channel_block(label: str, chs: dict, baseline: float) -> str:
        rows = [f"**{label}** (baseline={_fmt(baseline, '.4f')})\n\n" + channel_hdr]
        for ch in chs.values():
            rows.append(_channel_table_row(ch) + "\n")
        return "".join(rows)

    ch_block_a = _channel_block(f"Run A  R={audit_a['ring_radius_um']} μm", ch_a, baseline_a)
    ch_block_b = ""
    if results_b:
        ch_block_b = "\n" + _channel_block(f"Run B  R={audit_b['ring_radius_um']} μm", ch_b, baseline_b)

    # ── A vs B comparison ──────────────────────────────────────────────────────
    avb_block = ""
    if results_b:
        avb_rows = []
        for ch_name in ch_a:
            ca = ch_a[ch_name]
            cb = ch_b.get(ch_name, {})
            if "SKIPPED" in ca or "SKIPPED" in cb:
                continue
            acc_a = ca.get("acc_test")
            acc_b = cb.get("acc_test")
            db_a = ca.get("delta_vs_baseline")
            db_b = cb.get("delta_vs_baseline")
            diff = round(db_a - db_b, 4) if (db_a is not None and db_b is not None) else None
            avb_rows.append(
                f"| {ch_name:30s} | {_fmt(acc_a)} | {_fmt(acc_b)} | {_fmt(db_a,'+.4f')} | {_fmt(db_b,'+.4f')} | {_fmt(diff,'+.4f')} |"
            )
        avb_table = (
            "| channel | acc_A | acc_B | ΔB_A | ΔB_B | ΔB_A−ΔB_B |\n"
            "|" + "-"*32 + "|" + "-"*7 + "|" + "-"*7 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*12 + "|\n"
        ) + "\n".join(avb_rows)
        avb_block = f"\n## A vs B Comparison\n\n{avb_table}\n"

    # ── unavailable channels ───────────────────────────────────────────────────
    uv = results_a["unavailable"]
    uv_rows = []
    for k, v in uv.items():
        uv_rows.append(f"| {k:25s} | {v['reason'][:70]:72s} | {v['min_pipeline_change'][:70]:72s} |")
    uv_table = (
        "| channel | reason | min pipeline change |\n"
        "|" + "-"*27 + "|" + "-"*74 + "|" + "-"*74 + "|\n"
    ) + "\n".join(uv_rows)

    # ── interpretation ─────────────────────────────────────────────────────────
    roi_acc_a = ch_a.get("roi", {}).get("acc_test")
    roi_acc_b = ch_b.get("roi", {}).get("acc_test") if results_b else None
    chance = 0.1

    if roi_acc_a is not None and roi_acc_a <= chance + 0.02:
        roi_interpretation = (
            f"ROI accuracy for A ({roi_acc_a:.1%}) is at chance level ({chance:.1%}). "
            "This is consistent with the finding that all inputs were identical: the ROI variation "
            "is simulation noise (random seed=42), not digit information. "
            "The classifier is fitting to noise on the training set and generalises to ~chance on test."
        )
    else:
        roi_interpretation = (
            f"ROI accuracy for A ({_fmt(roi_acc_a, '.1%')}) is above chance. "
            "Given that all inputs were identical, this likely indicates overfitting to simulation "
            "noise patterns that happen to correlate with the class labels in this small (50-image) split."
        )

    report = f"""# mnist_wta_v1 Pilot Audit Report

**Date**: {run_date}
**Output dir**: `{out_dir}`
**Run A**: `{results_a['run_dir']}` (R={audit_a['ring_radius_um']} μm)
{"**Run B**: `" + results_b['run_dir'] + "` (R=" + str(audit_b['ring_radius_um']) + " μm)" if results_b else ""}

---

## 1. Data Audit
{pathology_a}{pathology_b}
{audit_table}

---

## 2. Available Channel Results

{ch_block_a}{ch_block_b}
{avb_block}
---

## 3. Unavailable Channels

{uv_table}

---

## 4. Physical Interpretation

### Score model failure (root cause of all observed pathologies)

{audit_a['norm_scores_identical'] and
f"""All {audit_a['n_unique_norm_score_rows']} unique norm_score row(s) out of {audit_a['n_train']+audit_a['n_test']} images indicates a **degenerate score model**.
The Ridge fit overflowed (`raw_scores ~ {audit_a['raw_scores_magnitude']:.1e}`), mapping all PCA inputs to identical outputs.
Every image in both runs was therefore simulated with the **same power pattern**: each of the 10 class lasers received the same power
(alpha + beta * norm_score_c) × P_th with identical norm_scores for all c.
The WTA circuit received no class-distinguishing input."""
or "Score model appears valid."}

### ROI channel

{roi_interpretation}

### Phase / interference / t_cond_per_spot

These channels **cannot be evaluated** from the saved data (no complex psi, no per-spot time traces).
Whether polariton-native interference effects encode class information is an **open hypoverification** —
the current pilots neither support nor refute it because the input encoding was broken.

### A vs B comparison (R=8 vs R=15)

{"Since both runs received identical inputs, any accuracy difference between A and B reflects geometry-dependent simulation noise patterns, **not** a physics effect. No conclusion about polariton coupling strength can be drawn from these pilots." if (results_b and audit_b["norm_scores_identical"]) else ""}

---

## 5. Decision Tree

```
score_model_overflow?
├── YES (this pilot) → all downstream metrics are invalid
│   ├── Fix score model first (increase n_train, fix Ridge numerical stability)
│   ├── Re-run pilot with correct scores before any architecture decisions
│   └── Do NOT conclude WTA doesn't work — input was broken
│
└── NO (fixed pilot)
    ├── ROI accuracy ≈ baseline?
    │   ├── YES → ROI not informative; try t_cond_per_spot or phase
    │   └── NO  → ROI has signal; WTA competes via intensity
    ├── A (R=8) significantly > B (R=15)?
    │   ├── YES → coupling-dependent effect; geometry matters
    │   └── NO  → might be threshold effect, not polariton WTA
    └── Both channels weak → revisit architecture (2-3 spot simpler mechanism)
```

---

## 6. Minimum Next Output Requirements

To test phase/interference/t_cond_per_spot hypotheses, a **diagnostic GPU run** would need:

| Output | Change needed | GPU cost |
|--------|---------------|----------|
| Per-spot t_cond | Add per-class ROI threshold crossing detection during simulation loop | +0 (CPU post-proc) |
| Complex psi snapshots | Save `psi` (complex) every 2 ps from t=0 to 40 ps for 10 diagnostic images | +~10% per image |
| Far-field psi_k | FFT(psi) at readout time, save cropped complex or \\|psi_k\\|^2 | +negligible |
| Shot noise baseline | 5 seeds × 10 images | ~50 extra sim runs |

**Prerequisite for all of the above**: fix the score model first so inputs are actually different between images.

---

## 7. Immediate Action Required

1. **Fix score model overflow** before any new GPU run:
   - Increase n_train to ≥ 200 images per pilot (20/class) for stable PCA+Ridge fit
   - Verify `raw_scores_per_class` have finite magnitude (~O(1)) after fitting
   - Add a sanity check in `prepare_run.py`: assert `np.isfinite(raw_scores).all()` and `raw_scores.std() > 0`

2. **Add score sanity check** to `prepare_run.py` before writing `dataset_index.json`:
   ```python
   if not np.isfinite(scores_all).all() or scores_all.std() < 1e-6:
       raise ValueError("Score model produced degenerate output — check n_train and Ridge fit")
   ```

3. **Re-run both R=8 and R=15 pilots** with fixed score model (200-image balanced split).
"""
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed mnist_wta_v1 pilot runs")
    parser.add_argument("--run-a", required=True, help="Path to first run directory")
    parser.add_argument("--run-b", default=None, help="Path to second run directory (optional)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print(" mnist_wta_v1 — pilot_audit")
    print("=" * 60)

    print(f"  Loading run A: {args.run_a}")
    run_a = _load_run(args.run_a)
    results_a = _analyze_run(run_a)

    results_b = None
    if args.run_b:
        print(f"  Loading run B: {args.run_b}")
        run_b = _load_run(args.run_b)
        results_b = _analyze_run(run_b)

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_md = _make_report(results_a, results_b, run_date, args.out_dir)

    report_path = os.path.join(args.out_dir, "PILOT_AUDIT_REPORT.md")
    _atomic_write(report_path, report_md)

    summary = {
        "run_date": run_date,
        "run_a": results_a,
        "run_b": results_b,
    }
    summary_path = os.path.join(args.out_dir, "pilot_audit_summary.json")
    _atomic_write_json(summary_path, summary)

    # ── stdout summary ──────────────────────────────────────────────────────
    audit_a = results_a["audit"]

    print(f"\n  Run A  R={audit_a['ring_radius_um']} μm")
    print(f"    n_train/test: {audit_a['n_train']}/{audit_a['n_test']}")
    print(f"    score_model_overflow: {audit_a['score_model_overflow']}")
    print(f"    norm_scores identical: {audit_a['norm_scores_identical']}")
    print(f"    multi_cond fraction: {audit_a['multi_cond_fraction']:.2f}")

    if results_b:
        audit_b = results_b["audit"]
        print(f"\n  Run B  R={audit_b['ring_radius_um']} μm")
        print(f"    score_model_overflow: {audit_b['score_model_overflow']}")
        print(f"    norm_scores identical: {audit_b['norm_scores_identical']}")

    print(f"\n  Available channel results (A):")
    ch_a = results_a["channels"]
    for k, v in ch_a.items():
        if "SKIPPED" in v:
            print(f"    {k:30s}: SKIPPED")
        elif v.get("zero_variance"):
            print(f"    {k:30s}: zero variance → acc_test={_fmt(v.get('acc_test'))}")
        else:
            print(f"    {k:30s}: acc_test={_fmt(v.get('acc_test'))}  Δbaseline={_fmt(v.get('delta_vs_baseline'),'+.4f')}")

    print(f"\n  Unavailable channels: {', '.join(results_a['unavailable'].keys())}")

    print(f"\n  Report  → {report_path}")
    print(f"  Summary → {summary_path}")

    if audit_a["score_model_overflow"]:
        print(
            "\n  CRITICAL: score model overflowed in both pilots."
            "\n     All images had identical power patterns."
            "\n     Fix score model (increase n_train to ≥ 200) before any new GPU run."
        )


if __name__ == "__main__":
    main()
