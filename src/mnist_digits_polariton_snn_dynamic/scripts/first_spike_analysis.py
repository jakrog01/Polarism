"""First-spike temporal analysis for the dynamic SNN MNIST pipeline.

Reads a completed run directory (traces_psi.npy + trace_times_ps.npy + labels.npy
+ encoded_powers.npy + input_images.npy) and evaluates whether the FIRST-SPIKE
TIME per readout channel alone is enough to classify MNIST digits.

Three (threshold, windowing) variants are compared:
  A. rel30_pulse    : threshold = 0.30 * per-channel peak, pulse-locked windows
  B. abs95_pulse    : threshold = 95th percentile of |psi|^2 across (N, T, K),
                      pulse-locked windows
  C. rel30_quantile : threshold = rel30, windows = 20 quantiles of t_first

For each variant we run:
  * PER-SPOT classification (49 independent NearestCentroid classifiers on
    scalar t_first(:, k) -> class), plotting accuracy as a 7x7 heatmap.
  * GLOBAL classification (LogisticRegression on the 50-D vector t_first(:, :))
    with confusion matrix, compared to the 0.855 baseline from the main pipeline.

Windows do NOT have to match the class count; pulse-locked variants use
10 pulse bins + 1 tail bin + 1 no-spike bin = 12 bins, quantile variant
uses 20 quantile bins + 1 no-spike bin = 21 bins.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PULSE_SIGMA = 1.5
PULSE_SEP = 10.0
N_PULSES = 10
CUTOFF_SIGMA = 3.0
PHASE = CUTOFF_SIGMA * PULSE_SIGMA
PULSE_CENTERS = PHASE + np.arange(N_PULSES, dtype=np.float64) * PULSE_SEP
POST_PULSE_END = 200.0
N_SPOTS = 49
N_CLASSES = 10
BASELINE_SUMMARY_TEST_ACC = 0.855


@dataclass(frozen=True)
class SpikeResult:
    """First-spike time matrix and its binning for one variant."""

    variant: str
    t_first: np.ndarray
    bins: np.ndarray
    edges: np.ndarray
    n_bins: int
    threshold_kind: str
    threshold_value: float


def compute_first_spike_relative(
    field: np.ndarray, times_ps: np.ndarray, alpha: float, t_max: float, abs_floor: float
) -> np.ndarray:
    peaks = field.max(axis=1)
    thresholds = alpha * peaks[:, None, :]
    above = field > thresholds
    any_above = above.any(axis=1)
    active = peaks > abs_floor
    valid = any_above & active
    first_idx = np.argmax(above, axis=1)
    return np.where(valid, times_ps[first_idx], np.float64(t_max))


def compute_first_spike_absolute(
    field: np.ndarray, times_ps: np.ndarray, threshold: float, t_max: float
) -> np.ndarray:
    above = field > threshold
    any_above = above.any(axis=1)
    first_idx = np.argmax(above, axis=1)
    return np.where(any_above, times_ps[first_idx], np.float64(t_max))


def make_pulse_edges(t_max: float) -> np.ndarray:
    left = PULSE_CENTERS - PULSE_SEP / 2.0
    right_last = PULSE_CENTERS[-1] + PULSE_SEP / 2.0
    return np.concatenate(
        [left, [right_last, POST_PULSE_END, t_max + 1.0]]
    ).astype(np.float64)


def make_quantile_edges(
    t_first: np.ndarray, n_bins: int, t_max: float
) -> np.ndarray:
    spiked = t_first[t_first < t_max]
    if spiked.size == 0:
        raise ValueError("no spikes detected across all channels")
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(spiked, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        v = float(edges[0])
        edges = np.array([v - 0.5, v + 0.5], dtype=np.float64)
    return np.concatenate([edges, [t_max + 1.0]]).astype(np.float64)


def bin_times(t_first: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = edges.size - 1
    idx = np.digitize(t_first, edges) - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


def build_variants(
    field: np.ndarray, times_ps: np.ndarray, t_max: float, n_quantile_bins: int
) -> dict[str, SpikeResult]:
    abs_floor = 1e-4 * float(field.max())
    t_rel = compute_first_spike_relative(field, times_ps, 0.30, t_max, abs_floor)
    thr_abs = float(np.quantile(field, 0.95))
    t_abs = compute_first_spike_absolute(field, times_ps, thr_abs, t_max)
    edges_pulse = make_pulse_edges(t_max)
    edges_quant = make_quantile_edges(t_rel, n_quantile_bins, t_max)
    return {
        "rel30_pulse": SpikeResult(
            variant="rel30_pulse",
            t_first=t_rel,
            bins=bin_times(t_rel, edges_pulse),
            edges=edges_pulse,
            n_bins=edges_pulse.size - 1,
            threshold_kind="relative",
            threshold_value=0.30,
        ),
        "abs95_pulse": SpikeResult(
            variant="abs95_pulse",
            t_first=t_abs,
            bins=bin_times(t_abs, edges_pulse),
            edges=edges_pulse,
            n_bins=edges_pulse.size - 1,
            threshold_kind="absolute_q95",
            threshold_value=thr_abs,
        ),
        "rel30_quantile": SpikeResult(
            variant="rel30_quantile",
            t_first=t_rel,
            bins=bin_times(t_rel, edges_quant),
            edges=edges_quant,
            n_bins=edges_quant.size - 1,
            threshold_kind="relative",
            threshold_value=0.30,
        ),
    }


def per_spot_classification(
    t_first: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = t_first.shape[0]
    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels
    )
    per_spot = np.zeros(N_SPOTS, dtype=np.float64)
    for k in range(N_SPOTS):
        x_tr = t_first[tr, k]
        x_te = t_first[te, k]
        if x_tr.std() < 1e-12:
            majority = int(np.bincount(labels[tr]).argmax())
            per_spot[k] = float(np.mean(labels[te] == majority))
            continue
        try:
            clf = NearestCentroid()
            clf.fit(x_tr.reshape(-1, 1), labels[tr])
            per_spot[k] = accuracy_score(labels[te], clf.predict(x_te.reshape(-1, 1)))
        except ValueError:
            majority = int(np.bincount(labels[tr]).argmax())
            per_spot[k] = float(np.mean(labels[te] == majority))
    return per_spot, tr, te


def global_classification(
    t_first: np.ndarray, labels: np.ndarray, seed: int, test_frac: float
) -> dict[str, float | np.ndarray]:
    n = t_first.shape[0]
    idx = np.arange(n)
    tr, te = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, n_jobs=-1),
    )
    model.fit(t_first[tr], labels[tr])
    pred_te = model.predict(t_first[te])
    pred_tr = model.predict(t_first[tr])
    cm = confusion_matrix(labels[te], pred_te, labels=np.arange(N_CLASSES))
    return {
        "acc_train": float(accuracy_score(labels[tr], pred_tr)),
        "acc_test": float(accuracy_score(labels[te], pred_te)),
        "confusion": cm.astype(np.int64),
    }


def _shade_pulse_windows(ax: plt.Axes, edges: np.ndarray, y0: float, y1: float) -> None:
    for k in range(edges.size - 1):
        if k % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (edges[k], y0),
                    edges[k + 1] - edges[k],
                    y1 - y0,
                    facecolor="0.85",
                    edgecolor="none",
                    zorder=0,
                )
            )
    for tc in PULSE_CENTERS:
        ax.axvline(tc, color="tab:red", ls=":", lw=0.8, alpha=0.6, zorder=1)


def plot_histograms(
    variants: dict[str, SpikeResult], t_max: float, out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), constrained_layout=True)
    for ax, (name, res) in zip(axes, variants.items()):
        t = res.t_first.ravel()
        hist, edges = np.histogram(t, bins=200, range=(0.0, t_max + 5.0))
        centers = 0.5 * (edges[:-1] + edges[1:])
        _shade_pulse_windows(ax, res.edges, 0.0, hist.max() * 1.05)
        ax.bar(centers, hist, width=(edges[1] - edges[0]), color="tab:blue", alpha=0.9)
        ax.set_xlim(-2.0, t_max + 5.0)
        ax.set_ylim(0.0, hist.max() * 1.10)
        ax.set_xlabel("first-spike time [ps]")
        ax.set_ylabel("count (all samples x channels)")
        ax.set_title(
            f"{name}\n"
            f"thr={res.threshold_kind}({res.threshold_value:.3g})  "
            f"n_bins={res.n_bins}"
        )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_raster_per_class(
    variant: SpikeResult, labels: np.ndarray, out_path: Path, samples_per_class: int = 5
) -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 7.0), constrained_layout=True, sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        pool = np.where(labels == c)[0]
        chosen = rng.choice(pool, size=min(samples_per_class, pool.size), replace=False)
        _shade_pulse_windows(ax, variant.edges, -0.5, N_SPOTS - 0.5)
        for si, sample_idx in enumerate(chosen):
            t_row = variant.t_first[sample_idx, :N_SPOTS]
            spiked = t_row < POST_PULSE_END
            ax.scatter(
                t_row[spiked],
                np.arange(N_SPOTS)[spiked],
                s=12,
                marker="|",
                color=plt.cm.tab10(si),
                alpha=0.85,
                label=f"sample {int(sample_idx)}" if c == 0 else None,
            )
        ax.set_title(f"class {c}", fontsize=10)
        ax.set_xlim(-2.0, POST_PULSE_END + 2.0)
        ax.set_ylim(-0.5, N_SPOTS - 0.5)
        if c >= 5:
            ax.set_xlabel("t [ps]")
        if c % 5 == 0:
            ax.set_ylabel("spot idx")
    fig.suptitle(f"first-spike raster per class  ({variant.variant})", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_spot_heatmap(
    per_spot_by_variant: dict[str, np.ndarray], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), constrained_layout=True)
    vmin = min(a.min() for a in per_spot_by_variant.values())
    vmax = max(a.max() for a in per_spot_by_variant.values())
    for ax, (name, acc) in zip(axes, per_spot_by_variant.items()):
        grid = acc.reshape(7, 7)
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
        for i in range(7):
            for j in range(7):
                ax.text(
                    j,
                    i,
                    f"{grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if grid[i, j] < 0.5 * (vmin + vmax) else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_title(f"{name}\nmean={acc.mean():.3f}  max={acc.max():.3f}")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="test accuracy (NearestCentroid)")
    fig.suptitle("per-spot classification accuracy (7x7 lattice, 1D feature = t_first)", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_confusions(
    global_by_variant: dict[str, dict], out_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), constrained_layout=True)
    for ax, (name, res) in zip(axes, global_by_variant.items()):
        cm = res["confusion"].astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.0%}" if cm_norm[i, j] > 0 else "",
                    ha="center",
                    va="center",
                    color="white" if cm_norm[i, j] < 0.5 else "black",
                    fontsize=6,
                )
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(
            f"{name}\nglobal LR: train={res['acc_train']:.3f}  test={res['acc_test']:.3f}"
        )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="row-normalized recall")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(
    global_by_variant: dict[str, dict],
    per_spot_by_variant: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    names = list(global_by_variant.keys())
    global_test = [global_by_variant[n]["acc_test"] for n in names]
    per_spot_max = [per_spot_by_variant[n].max() for n in names]
    per_spot_mean = [per_spot_by_variant[n].mean() for n in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    ax.bar(x - width, global_test, width=width, label="global LR test", color="tab:blue")
    ax.bar(x, per_spot_max, width=width, label="best single spot", color="tab:orange")
    ax.bar(x + width, per_spot_mean, width=width, label="mean over spots", color="tab:green")
    ax.axhline(BASELINE_SUMMARY_TEST_ACC, ls="--", color="black", lw=1.2, label=f"baseline summary features ({BASELINE_SUMMARY_TEST_ACC})")
    ax.axhline(1.0 / N_CLASSES, ls=":", color="gray", lw=1.0, label=f"chance ({1.0 / N_CLASSES:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("first-spike-only accuracy vs. baseline (600-D temporal summary features)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_traces(
    psi: np.ndarray,
    times_ps: np.ndarray,
    encoded_powers: np.ndarray,
    variant: SpikeResult,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.5), constrained_layout=True, sharex=True)
    n = psi.shape[0]
    sample_idx = 0
    mean_power = encoded_powers[sample_idx]
    bright_spot = int(np.argmax(mean_power))
    dim_spot = int(np.argmin(mean_power + (mean_power == 0) * 1e-9))
    picks = [
        (sample_idx, bright_spot, "brightest spot"),
        (sample_idx, dim_spot, "dimmest spot"),
        (sample_idx, N_SPOTS // 2, "central spot"),
        (sample_idx, N_SPOTS, "global CAP-clear channel"),
    ]
    for ax, (i, k, tag) in zip(axes.ravel(), picks):
        trace = psi[i, :, k]
        peak = trace.max()
        if variant.threshold_kind.startswith("relative"):
            thr = variant.threshold_value * peak
        else:
            thr = variant.threshold_value
        t1 = variant.t_first[i, k]
        _shade_pulse_windows(ax, variant.edges, 0.0, peak * 1.1)
        ax.plot(times_ps, trace, color="tab:blue", lw=1.0)
        ax.axhline(thr, color="tab:red", ls="--", lw=1.0, label=f"threshold={thr:.3g}")
        if t1 < POST_PULSE_END:
            ax.axvline(t1, color="tab:green", lw=1.2, label=f"t_first={t1:.2f} ps")
        else:
            ax.text(
                0.5, 0.9, "no spike", transform=ax.transAxes, ha="center",
                color="tab:red", fontsize=11,
            )
        ax.set_ylim(0.0, peak * 1.10 if peak > 0 else 1.0)
        ax.set_title(f"sample={i}, spot={k}  ({tag})  power={mean_power[k] if k < N_SPOTS else float('nan'):.1f}")
        ax.set_ylabel("|psi|^2 (mask avg)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("t [ps]")
    axes[-1, 1].set_xlabel("t [ps]")
    fig.suptitle(f"threshold + first-spike demo ({variant.variant})", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--field", choices=["psi", "nA", "nI"], default="psi",
                        help="which channel-trace file to analyze")
    parser.add_argument("--n-quantile-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    field_files = {"psi": "traces_psi.npy", "nA": "traces_nA.npy", "nI": "traces_nI.npy"}
    field_file = field_files[args.field]
    print(f"[load] {run_dir}  field={args.field} ({field_file})")
    field = np.load(run_dir / field_file).astype(np.float64, copy=False)
    times_ps = np.load(run_dir / "trace_times_ps.npy").astype(np.float64, copy=False)
    labels = np.load(run_dir / "labels.npy").astype(np.int64, copy=False)
    encoded_powers = np.load(run_dir / "encoded_powers.npy").astype(np.float64, copy=False)
    n_samples, n_frames, n_channels = field.shape
    print(f"       field shape = {field.shape}, times [{times_ps[0]:.2f}, {times_ps[-1]:.2f}] ps")
    print(f"       labels      = {np.bincount(labels)}")
    print(f"       field.max = {field.max():.3g}, field.min = {field.min():.3g}")

    t_max = float(times_ps[-1])
    variants = build_variants(field, times_ps, t_max, args.n_quantile_bins)
    for name, res in variants.items():
        n_no_spike = int(np.sum(res.t_first >= POST_PULSE_END))
        print(
            f"[variant {name:<16}] thr={res.threshold_kind}({res.threshold_value:.4g})  "
            f"n_bins={res.n_bins}  no_spike_frac={n_no_spike / res.t_first.size:.3f}"
        )

    per_spot_by_variant: dict[str, np.ndarray] = {}
    global_by_variant: dict[str, dict] = {}
    for name, res in variants.items():
        print(f"[classify {name}] per-spot ...")
        per_spot_acc, _, _ = per_spot_classification(
            res.t_first, labels, seed=args.seed, test_frac=args.test_fraction
        )
        per_spot_by_variant[name] = per_spot_acc
        print(f"                   mean={per_spot_acc.mean():.3f}  max={per_spot_acc.max():.3f}")
        print(f"[classify {name}] global ...")
        global_res = global_classification(
            res.t_first, labels, seed=args.seed, test_frac=args.test_fraction
        )
        global_by_variant[name] = global_res
        print(f"                   train={global_res['acc_train']:.3f}  test={global_res['acc_test']:.3f}")

    print("[plot] histograms ...")
    plot_histograms(variants, t_max, out_dir / "hist_first_spike.png")
    for name, res in variants.items():
        plot_raster_per_class(res, labels, out_dir / f"raster_{name}.png")
    print("[plot] per-spot heatmap ...")
    plot_per_spot_heatmap(per_spot_by_variant, out_dir / "per_spot_accuracy.png")
    print("[plot] global confusion matrices ...")
    plot_global_confusions(global_by_variant, out_dir / "global_confusions.png")
    print("[plot] summary bars ...")
    plot_summary_bars(global_by_variant, per_spot_by_variant, out_dir / "summary_bars.png")
    print("[plot] example traces ...")
    plot_example_traces(
        field, times_ps, encoded_powers, variants["rel30_pulse"],
        out_dir / "example_traces_rel30_pulse.png",
    )
    plot_example_traces(
        field, times_ps, encoded_powers, variants["abs95_pulse"],
        out_dir / "example_traces_abs95_pulse.png",
    )

    summary = {
        "run_dir": str(run_dir),
        "field": args.field,
        "field_max": float(field.max()),
        "n_samples": int(n_samples),
        "n_frames": int(n_frames),
        "n_channels": int(n_channels),
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
        "baseline_summary_test_acc": BASELINE_SUMMARY_TEST_ACC,
        "chance_level": 1.0 / N_CLASSES,
        "pulse_centers_ps": PULSE_CENTERS.tolist(),
        "post_pulse_end_ps": POST_PULSE_END,
        "variants": {
            name: {
                "threshold_kind": res.threshold_kind,
                "threshold_value": float(res.threshold_value),
                "n_bins": int(res.n_bins),
                "edges_ps": _json_ready(res.edges),
                "no_spike_fraction": float(
                    np.sum(res.t_first >= POST_PULSE_END) / res.t_first.size
                ),
                "per_spot_accuracy_mean": float(per_spot_by_variant[name].mean()),
                "per_spot_accuracy_max": float(per_spot_by_variant[name].max()),
                "per_spot_accuracy_argmax": int(per_spot_by_variant[name].argmax()),
                "per_spot_accuracy_min": float(per_spot_by_variant[name].min()),
                "per_spot_accuracy_full": _json_ready(per_spot_by_variant[name]),
                "global_acc_train": float(global_by_variant[name]["acc_train"]),
                "global_acc_test": float(global_by_variant[name]["acc_test"]),
                "global_confusion": _json_ready(global_by_variant[name]["confusion"]),
            }
            for name, res in variants.items()
        },
    }
    with open(out_dir / "first_spike_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] summary + plots in {out_dir}")


if __name__ == "__main__":
    main()
