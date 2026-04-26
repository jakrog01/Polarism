"""CPU Stage 4: finalise.

Aggregates per-image metadata into ``results_summary.json`` and generates
cross-image comparison plots from scalar sidecars and trace comparison PNGs.

Pipeline order
--------------
1. ``prepare_reference``  — MNIST encode + ODE target traces (per image)
2. ``fit_dot_size``       — GPU global sigma_space search → ``fit_result.json``
3. ``run_scenario``       — GPU full spatial run per image → per-image artifacts
4. ``finalize``           — this stage → ``results_summary.json``

Invoked as::

    python -m dot_response_fit.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dot_response_fit.config.loader import load_config
from dot_response_fit.manifest.io import (
    atomic_write_json,
    set_manifest_field,
)

PLOT_DPI = 200


def _load_image_meta(image_dir: str) -> dict | None:
    meta_path = os.path.join(image_dir, "image_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _generate_summary_plot(
    image_results: list[dict],
    fit_result: dict,
    results_dir: str,
) -> str:
    """Generate aggregate score vs sigma_space and per-image RMSE table."""
    sigma_values = [r["sigma_space"] for r in fit_result.get("all_results", [])]
    mean_rmse = [
        r["mean_rmse"] if r["mean_rmse"] is not None else float("nan")
        for r in fit_result.get("all_results", [])
    ]
    best_sigma = fit_result.get("best_sigma_space")

    n_images = len(image_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax_score = axes[0]
    if sigma_values and any(not np.isnan(v) for v in mean_rmse):
        ax_score.plot(sigma_values, mean_rmse, "o-", color="steelblue", lw=1.8)
        if best_sigma is not None:
            ax_score.axvline(best_sigma, color="darkorange", ls="--", lw=1.5, label=f"best={best_sigma:.1f}")
            ax_score.legend(fontsize=9)
    ax_score.set_xlabel("sigma_space (µm)")
    ax_score.set_ylabel("mean RMSE (normalised)")
    ax_score.set_title("Aggregate score vs sigma_space")
    ax_score.grid(True, alpha=0.3)

    ax_per = axes[1]
    if image_results:
        labels = [r["image_id"] for r in image_results]
        rmse_vals = [
            r.get("fit_score") if r.get("fit_score") is not None else float("nan")
            for r in image_results
        ]
        x = np.arange(len(labels))
        colors = ["steelblue" if not np.isnan(v) else "lightcoral" for v in rmse_vals]
        ax_per.bar(x, [v if not np.isnan(v) else 0 for v in rmse_vals], color=colors, width=0.6)
        ax_per.set_xticks(x)
        ax_per.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax_per.set_ylabel("RMSE")
        ax_per.set_title(f"Per-image RMSE  (sigma_space={best_sigma} µm)")
        ax_per.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Dot-Response Fit Summary  |  n_images={n_images}  |  best_sigma={best_sigma} µm",
        fontsize=12,
        fontweight="bold",
    )
    out_path = os.path.join(results_dir, "summary.png")
    os.makedirs(results_dir, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU Stage 4: finalise and summarise")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    print("=" * 60)
    print(" CPU Finalize")
    print("=" * 60)
    print(f"  Run dir   : {run_dir}")

    index_path = os.path.join(run_dir, "reference", "images", "index.json")
    if not os.path.isfile(index_path):
        print(f"ERROR: reference index not found: {index_path}", file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        image_index: list[dict] = json.load(f)

    fit_path = os.path.join(run_dir, "fit_result.json")
    fit_result: dict = {}
    if os.path.isfile(fit_path):
        with open(fit_path) as f:
            fit_result = json.load(f)

    best_sigma = fit_result.get("best_sigma_space")

    results_images_dir = os.path.join(run_dir, "results", "images")
    image_results: list[dict] = []
    missing: list[str] = []

    for entry in image_index:
        image_id = entry["image_id"]
        image_dir = os.path.join(results_images_dir, image_id)
        meta = _load_image_meta(image_dir)
        if meta is None:
            missing.append(image_id)
            print(f"  WARNING: missing image_meta.json for {image_id}", file=sys.stderr)
            continue

        scalars_path = os.path.join(image_dir, "scalars.npz")
        trace_cmp_path = os.path.join(image_dir, "trace_comparison.png")

        image_results.append({
            "image_id": image_id,
            "dataset_index": entry["dataset_index"],
            "digit_class": entry["digit_class"],
            "n_pixels": entry["n_pixels"],
            "sigma_space": meta.get("sigma_space"),
            "fit_score": meta.get("fit_score"),
            "t_cond": meta.get("t_cond"),
            "total_sim_time": meta.get("total_sim_time"),
            "paths": {
                "image_meta": os.path.relpath(os.path.join(image_dir, "image_meta.json"), run_dir),
                "trace_comparison": os.path.relpath(trace_cmp_path, run_dir) if os.path.isfile(trace_cmp_path) else None,
                "scalars": os.path.relpath(scalars_path, run_dir) if os.path.isfile(scalars_path) else None,
            },
        })
        print(
            f"  {image_id}: RMSE={'%.6f' % meta['fit_score'] if meta.get('fit_score') else 'N/A'}"
            f"  digit={entry['digit_class']}  t_cond={meta.get('t_cond')}"
        )

    fit_section = {
        "best_sigma_space": fit_result.get("best_sigma_space"),
        "best_score": fit_result.get("best_score"),
        "observable": fit_result.get("observable", "psi_sq_max"),
        "aggregate_method": fit_result.get("aggregate_method", "mean_rmse"),
        "n_fit_pixels": fit_result.get("n_fit_pixels"),
        "fit_reference_source": fit_result.get("fit_reference_source"),
        "search_completed": fit_result.get("search_completed"),
        "n_images_in_fit": fit_result.get("n_images"),
        "all_results": fit_result.get("all_results", []),
    }

    summary = {
        "run_dir": run_dir,
        "n_images_reference": len(image_index),
        "n_images_completed": len(image_results),
        "n_images_missing": len(missing),
        "missing_image_ids": missing,
        "fit": fit_section,
        "images": image_results,
    }
    summary_path = os.path.join(run_dir, "results_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    print("  Generating summary plot ...")
    try:
        plot_path = _generate_summary_plot(image_results, fit_result, results_dir)
        print(f"    {plot_path}")
    except Exception as e:
        print(f"  WARNING: summary plot failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "summary_path", summary_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print(f"\n  Finalize complete.  ({len(image_results)}/{len(image_index)} images)")
    if missing:
        print(f"  WARNING: {len(missing)} image(s) missing results.", file=sys.stderr)


if __name__ == "__main__":
    main()
