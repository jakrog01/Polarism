"""CPU stage: summarize spacetime mechanism traces and generate diagnostic plots."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
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


def _atomic_write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _selected_roi_indices(roi_ids: np.ndarray) -> list[int]:
    ids = [str(v) for v in roi_ids]
    selected = []
    for i, roi_id in enumerate(ids):
        if roi_id in {"center", "mixer"} or roi_id.startswith("out_"):
            selected.append(i)
    if not selected:
        selected = list(range(min(len(ids), 12)))
    return selected[:16]


def _plot_layout(scenario: dict[str, Any], out_path: str) -> None:
    if not _MPL:
        return
    role_colors = {
        "feature": "#2f6fbd",
        "read_probe": "#d1495b",
        "input": "#2a9d8f",
        "mixer_gate": "#f4a261",
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    for laser in scenario.get("lasers", []):
        role = str(laser.get("role", "pump"))
        power = max(float(laser.get("power", 0.0)), 0.0)
        size = 30.0 + 90.0 * min(power / 2500.0, 2.5)
        ax.scatter(
            [float(laser["x0"])],
            [float(laser["y0"])],
            s=size,
            color=role_colors.get(role, "#555555"),
            alpha=0.85 if power > 0 else 0.22,
            edgecolor="black",
            linewidth=0.4,
            label=role,
        )
        ax.text(float(laser["x0"]), float(laser["y0"]), str(laser["id"]), fontsize=7)

    for roi in scenario.get("rois", []):
        circ = plt.Circle(
            (float(roi["x0"]), float(roi["y0"])),
            float(roi["radius"]),
            fill=False,
            linewidth=0.8,
            linestyle="--",
            color="#333333",
            alpha=0.6,
        )
        ax.add_patch(circ)
        if str(roi["id"]).startswith("out_") or str(roi["id"]) in {"center", "mixer"}:
            ax.text(float(roi["x0"]), float(roi["y0"]), str(roi["id"]), fontsize=7, color="#111111")

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        unique.setdefault(label, h)
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(scenario["name"])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_scalar_traces(data: Any, out_path: str, title: str) -> None:
    if not _MPL:
        return
    t = data["time_ps"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    series = [
        ("psi_sq_max", "max |psi|^2"),
        ("nR_max", "max nR"),
        ("nI_max", "max nI"),
        ("pump_max", "max pump"),
    ]
    for ax, (key, label) in zip(axes, series):
        ax.plot(t, data[key], linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[-2].set_xlabel("t [ps]")
    axes[-1].set_xlabel("t [ps]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_roi_traces(data: Any, out_path: str, title: str) -> None:
    if not _MPL:
        return
    roi_ids = np.asarray(data["roi_ids"]).astype(str)
    idxs = _selected_roi_indices(roi_ids)
    t = data["time_ps"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fields = [
        ("roi_psi_integrals", "ROI integral |psi|^2"),
        ("roi_nR_means", "ROI mean nR"),
        ("roi_emission_integrals", "ROI integral R nR |psi|^2"),
    ]
    for ax, (key, ylabel) in zip(axes, fields):
        arr = data[key]
        for idx in idxs:
            ax.plot(t, arr[:, idx], linewidth=1.2, label=str(roi_ids[idx]))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("t [ps]")
    axes[0].legend(loc="upper right", ncol=4, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_final_fields(data: Any, out_path: str, title: str) -> None:
    if not _MPL:
        return
    keys = [key for key in ("final_psi_sq_ds", "final_nR_ds", "final_nI_ds") if key in data.files]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        im = ax.imshow(data[key], origin="lower", cmap="magma")
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _summarize_scenario(meta: dict[str, Any], data: Any) -> dict[str, Any]:
    roi_ids = np.asarray(data["roi_ids"]).astype(str)
    final_integrals = np.asarray(data["roi_psi_integrals"][-1], dtype=np.float64)
    peak_integrals = np.max(np.asarray(data["roi_psi_integrals"], dtype=np.float64), axis=0)

    output_idxs = [i for i, roi_id in enumerate(roi_ids) if str(roi_id).startswith("out_")]
    if output_idxs:
        out_peaks = peak_integrals[output_idxs]
        winner_local = int(np.argmax(out_peaks))
        winner_idx = output_idxs[winner_local]
        sorted_peaks = np.sort(out_peaks)[::-1]
        out_margin = float(sorted_peaks[0] / max(sorted_peaks[1], 1e-30)) if len(sorted_peaks) > 1 else None
        winner_output = str(roi_ids[winner_idx])
    else:
        out_margin = None
        winner_output = None

    row: dict[str, Any] = {
        "name": meta["scenario"]["name"],
        "architecture": meta["scenario"]["architecture"],
        "pattern": meta["scenario"].get("pattern"),
        "condensed": meta.get("condensed"),
        "t_cond_ps": meta.get("t_cond_ps"),
        "psi_sq_max_peak": meta.get("psi_sq_max_peak"),
        "nR_max_peak": meta.get("nR_max_peak"),
        "nI_max_peak": meta.get("nI_max_peak"),
        "pump_dose": meta.get("pump_dose"),
        "elapsed_s": meta.get("elapsed_s"),
        "winner_output_peak": winner_output,
        "output_peak_margin": out_margin,
    }
    for roi_id in ("center", "mixer"):
        matches = np.where(roi_ids == roi_id)[0]
        if len(matches):
            idx = int(matches[0])
            row[f"{roi_id}_final_psi_integral"] = float(final_integrals[idx])
            row[f"{roi_id}_peak_psi_integral"] = float(peak_integrals[idx])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize mnist_spacetime_v1 scenario campaign")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(" mnist_spacetime_v1 - finalize")
    print("=" * 70)

    meta_dir = os.path.join(run_dir, "metadata")
    trace_dir = os.path.join(run_dir, "traces")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    meta_files = sorted(
        f for f in os.listdir(meta_dir)
        if f.endswith(".json") and f != "campaign_gpu_summary.json"
    )
    if not meta_files:
        print("ERROR: no scenario metadata found", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fname in meta_files:
        meta = _load_json(os.path.join(meta_dir, fname))
        scenario = meta["scenario"]
        trace_path = os.path.join(run_dir, meta["trace_file"])
        if not os.path.isfile(trace_path):
            print(f"WARNING: missing trace for {scenario['name']}: {trace_path}", file=sys.stderr)
            continue
        data = np.load(trace_path, allow_pickle=True)
        rows.append(_summarize_scenario(meta, data))

        this_plot_dir = os.path.join(plot_dir, scenario["name"])
        os.makedirs(this_plot_dir, exist_ok=True)
        _plot_layout(scenario, os.path.join(this_plot_dir, "pump_layout.png"))
        _plot_scalar_traces(data, os.path.join(this_plot_dir, "scalar_traces.png"), scenario["name"])
        _plot_roi_traces(data, os.path.join(this_plot_dir, "roi_traces.png"), scenario["name"])
        _plot_final_fields(data, os.path.join(this_plot_dir, "final_fields_downsampled.png"), scenario["name"])

    if not rows:
        print("ERROR: no complete scenario traces found", file=sys.stderr)
        sys.exit(1)

    condensed_rate = float(np.mean([bool(row.get("condensed")) for row in rows]))
    summary = {
        "package": "mnist_spacetime_v1",
        "run_dir": run_dir,
        "n_scenarios": len(rows),
        "condensed_rate": condensed_rate,
        "scenarios": rows,
        "plots_dir": "plots",
        "summary_csv": "summary_table.csv",
    }
    atomic_write_json(os.path.join(run_dir, "results_summary_spacetime.json"), summary)
    _atomic_write_csv(os.path.join(run_dir, "summary_table.csv"), rows)

    print(f"  Scenarios      : {len(rows)}")
    print(f"  Condensed rate : {condensed_rate:.3f}")
    print(f"  Summary        : {run_dir}/results_summary_spacetime.json")
    print(f"  CSV            : {run_dir}/summary_table.csv")
    if _MPL:
        print(f"  Plots          : {run_dir}/plots/")
    else:
        print("  Plots          : skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
