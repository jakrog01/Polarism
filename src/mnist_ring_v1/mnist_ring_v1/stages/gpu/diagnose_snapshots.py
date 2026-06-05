"""GPU stage: spatial snapshot diagnostics for a small subset of images.

Selects 10-15 images (1-2 per digit + ambivalent examples), runs simulations
with periodic spatial checkpoints, and saves downsampled .npz per image plus
a diagnostic 3×3 angular-profile panel.

Invoked as:
    python -m mnist_ring_v1.stages.gpu.diagnose_snapshots \\
        --run-dir <run_dir> [--snapshot-interval-ps 10.0] [--max-time-ps 135.0]
        [--downsample-factor 8] [--n-images 15]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any

import numpy as np

from mnist_ring_v1.config.loader import get_encoding_cfg, get_readout_cfg, load_config
from mnist_ring_v1.encoding.geometry import ring_spot_positions
from mnist_ring_v1.encoding.mnist import load_mnist
from mnist_ring_v1.encoding.projection import PCAProjection
from mnist_ring_v1.simulation.batch_core import SharedSimResources
from mnist_ring_v1.simulation.lasers import build_ring_trigger_lasers
from mnist_ring_v1.simulation.snapshots import simulate_one_image_with_snapshots
from mnist_ring_v1.stages.gpu.run_batch import _atomic_write_json, _build_config
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters


def _select_snapshot_images(
    dataset_index: list[dict],
    x_test_all: np.ndarray,
    y_test_all: np.ndarray,
    pca: PCAProjection,
    f_min: np.ndarray,
    f_max: np.ndarray,
    max_total: int,
) -> list[dict]:
    """Select images for snapshot run.

    Strategy:
    - Up to 2 test images per digit for digits 0, 1, 4 (for the 3×3 panel).
    - 1 test image per other digit.
    - Fill up to max_total with the most ambivalent test images (lowest Ridge margin).
    """
    test_entries = [e for e in dataset_index if e["split"] == "test"]
    if not test_entries:
        test_entries = [e for e in dataset_index if e["split"] == "train"][:max_total]

    by_digit: dict[int, list[dict]] = {d: [] for d in range(10)}
    for e in test_entries:
        d = int(e.get("label", -1))
        if 0 <= d <= 9:
            by_digit[d].append(e)

    panel_digits = {0, 1, 4}
    selected_ids: set[str] = set()
    ordered: list[dict] = []

    for d in range(10):
        limit = 2 if d in panel_digits else 1
        for e in by_digit[d][:limit]:
            if e["image_id"] not in selected_ids:
                selected_ids.add(e["image_id"])
                ordered.append(e)

    if len(ordered) >= max_total:
        return ordered[:max_total]

    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler

        all_test_feats = []
        all_test_entries = []
        for e in test_entries:
            orig_idx = int(e["dataset_index"])
            img = x_test_all[orig_idx]
            feat_raw = pca.transform(img[np.newaxis, :])[0]
            feat_range = f_max - f_min
            feat_norm = (feat_raw - f_min) / np.where(feat_range > 1e-8, feat_range, 1.0)
            all_test_feats.append(feat_norm)
            all_test_entries.append(e)

        X_feat = np.stack(all_test_feats, axis=0)
        y_lbl = np.array([int(e["label"]) for e in all_test_entries])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_feat)
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X_scaled, y_lbl)
        scores = clf.decision_function(X_scaled)

        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        margins = sorted_scores[:, 0] - sorted_scores[:, 1]

        sorted_by_margin = np.argsort(margins)
        for idx in sorted_by_margin:
            e = all_test_entries[idx]
            if e["image_id"] not in selected_ids and len(ordered) < max_total:
                selected_ids.add(e["image_id"])
                ordered.append(e)
    except ImportError:
        remaining = [e for e in test_entries if e["image_id"] not in selected_ids]
        for e in remaining:
            if len(ordered) >= max_total:
                break
            selected_ids.add(e["image_id"])
            ordered.append(e)

    return ordered[:max_total]


def _save_angular_panel(
    snap_results: dict[str, Any],
    selected_entries: list[dict],
    target_time_ps: float,
    out_path: str,
    n_angular_bins: int = 64,
) -> None:
    """3×3 panel of n_R angular profiles at target_time_ps for digits 0, 1, 4."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — angular panel skipped")
        return

    panel_digits = [0, 1, 4]
    panel_examples: dict[int, list[str]] = {d: [] for d in panel_digits}
    for e in selected_entries:
        d = int(e.get("label", -1))
        if d in panel_examples and len(panel_examples[d]) < 3:
            panel_examples[d].append(e["image_id"])

    n_rows = 3
    n_cols = max(len(ids) for ids in panel_examples.values()) if any(panel_examples.values()) else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    theta_centers = np.linspace(-np.pi, np.pi, n_angular_bins + 1)[:-1] + np.pi / n_angular_bins

    all_vals = []
    for row_i, digit in enumerate(panel_digits):
        for col_j, img_id in enumerate(panel_examples[digit]):
            res = snap_results.get(img_id)
            if res is None:
                continue
            times = res.get("times_ps", np.array([]))
            profiles = res.get("ring_nR_angular_profile", None)
            if profiles is None or len(times) == 0:
                continue
            t_idx = int(np.argmin(np.abs(times - target_time_ps)))
            all_vals.append(profiles[t_idx])

    vmax = float(np.max([v.max() for v in all_vals])) if all_vals else 1.0
    vmax = max(vmax, 1e-10)

    for row_i, digit in enumerate(panel_digits):
        for col_j in range(n_cols):
            ax = axes[row_i, col_j]
            ax.set_ylim(0, vmax * 1.1)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"], fontsize=7)

            if col_j < len(panel_examples[digit]):
                img_id = panel_examples[digit][col_j]
                res = snap_results.get(img_id)
                if res is not None:
                    times = res.get("times_ps", np.array([]))
                    profiles = res.get("ring_nR_angular_profile", None)
                    if profiles is not None and len(times) > 0:
                        t_idx = int(np.argmin(np.abs(times - target_time_ps)))
                        ax.plot(theta_centers, profiles[t_idx], color="steelblue", lw=1.2)
                        ax.set_title(f"digit={digit}  {img_id}", fontsize=7)
            else:
                ax.set_visible(False)
                continue

            if col_j == 0:
                ax.set_ylabel("n_R  (a.u.)", fontsize=8)
            if row_i == n_rows - 1:
                ax.set_xlabel("θ (rad)", fontsize=8)

    fig.suptitle(
        f"Ring n_R angular profile at t={target_time_ps:.0f} ps  (digits 0, 1, 4)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Angular panel → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot diagnostics for mnist_ring_v1")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--snapshot-interval-ps", type=float, default=10.0)
    parser.add_argument("--max-time-ps", type=float, default=135.0)
    parser.add_argument("--downsample-factor", type=int, default=8)
    parser.add_argument("--n-images", type=int, default=15)
    parser.add_argument("--n-angular-bins", type=int, default=64)
    parser.add_argument("--angular-dr-um", type=float, default=2.0)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(run_dir, "config.yaml")) as f:
        import yaml
        cfg = yaml.safe_load(f)

    with open(os.path.join(run_dir, "dataset_index.json")) as f:
        dataset_index = json.load(f)

    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)

    calib_path = os.path.join(run_dir, "calibration_result.json")
    if not os.path.isfile(calib_path):
        print("ERROR: calibration_result.json not found — run calibration first.", file=sys.stderr)
        sys.exit(1)

    with open(calib_path) as f:
        calib_result = json.load(f)

    enc_cfg = get_encoding_cfg(cfg)
    readout_cfg = get_readout_cfg(cfg)

    sigma_time = float(enc_cfg["sigma_time_ps"])
    sigma_time_ring = float(enc_cfg.get("sigma_time_ring_ps", sigma_time))
    sigma_time_trigger = float(enc_cfg.get("sigma_time_trigger_ps", sigma_time))
    sigma_space = float(enc_cfg["sigma_space_um"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])
    ring_r = float(enc_cfg["ring_radius_um"])
    ring_rot = float(enc_cfg["ring_rotation_rad"])
    T_max = float(enc_cfg["T_max"])

    calib_cfg = cfg.get("calibration", {})
    ring_sub_frac = float(calib_cfg.get("P_ring_subthreshold_fraction", 0.8))
    trig_frac = float(calib_cfg.get("P_trigger_fraction", 0.8))

    ring_th_entry = calib_result.get("ring_only", {})
    trig_th_entry = calib_result.get("trigger_only", {})
    ring_th = ring_th_entry.get("estimated_threshold")
    trig_th = trig_th_entry.get("estimated_threshold")
    if ring_th is None or trig_th is None:
        print("ERROR: calibration thresholds not found in calibration_result.json", file=sys.stderr)
        sys.exit(1)

    guard = calib_result.get("pretrigger_guard", {})
    if not guard.get("skipped", True) and guard.get("recommended_ring_fraction") is not None:
        ring_sub_frac = float(guard["recommended_ring_fraction"])

    ring_power = float(ring_th) * ring_sub_frac
    trigger_power = float(trig_th) * trig_frac

    mnist_cfg = cfg["mnist"]
    x_tr, y_tr, x_te, y_te = load_mnist(mnist_cfg["data_path"])

    pca_path = os.path.join(run_dir, "projection_model.npz")
    norm_path = os.path.join(run_dir, "feature_normalizer.npz")
    pca = PCAProjection.load(pca_path)
    norm_data = np.load(norm_path)
    f_min = norm_data["f_min"]
    f_max = norm_data["f_max"]

    print("=" * 60)
    print(" mnist_ring_v1 — diagnose_snapshots")
    print("=" * 60)

    selected = _select_snapshot_images(
        dataset_index, x_te, y_te, pca, f_min, f_max, max_total=args.n_images
    )
    print(f"  Selected {len(selected)} images for snapshots")
    by_digit_count: dict[int, int] = {}
    for e in selected:
        d = int(e.get("label", -1))
        by_digit_count[d] = by_digit_count.get(d, 0) + 1
    print(f"  Digit distribution: {dict(sorted(by_digit_count.items()))}")

    snapshot_times = np.arange(0.0, args.max_time_ps + 0.5 * args.snapshot_interval_ps, args.snapshot_interval_ps)
    snapshot_times = snapshot_times[snapshot_times <= args.max_time_ps]
    print(f"  Snapshot times: {snapshot_times[0]:.0f}..{snapshot_times[-1]:.0f} ps  "
          f"({len(snapshot_times)} snapshots, every {args.snapshot_interval_ps:.0f} ps)")

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = _build_config(cfg)
    resources = SharedSimResources(sim_cfg)
    grid = resources.grid

    ring_xs, ring_ys = ring_spot_positions(ring_r, ring_rot)

    snaps_dir = os.path.join(run_dir, "snapshots")
    os.makedirs(snaps_dir, exist_ok=True)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    id_to_entry = {e["image_id"]: e for e in dataset_index}
    snap_results: dict[str, Any] = {}
    snapshot_index: list[dict] = []

    for img_i, sel_entry in enumerate(selected):
        img_id = sel_entry["image_id"]
        entry = id_to_entry[img_id]
        ring_delays = np.array(entry["ring_delays_ps"], dtype=np.float64)
        trig_delay_val = float(entry["trigger_delay_ps"])
        label = int(entry["label"])

        lasers = build_ring_trigger_lasers(
            ring_xs=ring_xs,
            ring_ys=ring_ys,
            ring_delays=ring_delays,
            ring_power=ring_power,
            trigger_delay=trig_delay_val,
            trigger_power=trigger_power,
            sigma_space_um=sigma_space,
            sigma_time_ring_ps=sigma_time_ring,
            sigma_time_trigger_ps=sigma_time_trigger,
            cutoff_sigma=cutoff,
            power_definition=power_def,
            grid_X=grid.X,
            grid_Y=grid.Y,
        )

        print(f"  [{img_i + 1}/{len(selected)}] {img_id}  label={label} ...", end=" ", flush=True)
        result = simulate_one_image_with_snapshots(
            resources=resources,
            lasers=lasers,
            snapshot_times_ps=snapshot_times,
            ring_radius_um=ring_r,
            downsample_factor=args.downsample_factor,
            n_angular_bins=args.n_angular_bins,
            angular_dr_um=args.angular_dr_um,
        )
        print(f"condensed={result['condensed']}  t_cond={result['t_cond']}")

        npz_path = os.path.join(snaps_dir, f"snap_{img_id}.npz")
        dir_ = os.path.dirname(os.path.abspath(npz_path))
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp.npz")
        os.close(fd)
        np.savez_compressed(
            tmp,
            times_ps=result["times_ps"],
            psi_sq_downsampled=result["psi_sq_downsampled"],
            nR_downsampled=result["nR_downsampled"],
            nI_downsampled=result["nI_downsampled"],
            pump_downsampled=result["pump_downsampled"],
            ring_nR_angular_profile=result["ring_nR_angular_profile"],
            image_id=np.array([img_id]),
            label=np.array([label]),
            ring_delays_ps=ring_delays,
            trigger_delay_ps=np.array([trig_delay_val]),
            ring_power=np.array([ring_power]),
            trigger_power=np.array([trigger_power]),
            sigma_time_ring_ps=np.array([sigma_time_ring]),
            sigma_time_trigger_ps=np.array([sigma_time_trigger]),
            downsample_factor=np.array([args.downsample_factor]),
            ring_radius_um=np.array([ring_r]),
            n_angular_bins=np.array([args.n_angular_bins]),
        )
        os.replace(tmp, npz_path)

        snap_results[img_id] = result
        snapshot_index.append({
            "image_id": img_id,
            "label": label,
            "split": entry["split"],
            "npz_path": os.path.relpath(npz_path, run_dir),
            "condensed": bool(result["condensed"]),
            "t_cond_ps": result["t_cond"],
            "n_snapshots": int(len(result["times_ps"])),
        })

    panel_time_ps = min(100.0, float(snapshot_times[-1]))
    panel_path = os.path.join(plots_dir, "nR_angular_panel.png")
    _save_angular_panel(
        snap_results,
        selected,
        panel_time_ps,
        panel_path,
        n_angular_bins=args.n_angular_bins,
    )

    _atomic_write_json(
        os.path.join(run_dir, "snapshot_index.json"),
        {
            "n_images": len(selected),
            "snapshot_times_ps": snapshot_times.tolist(),
            "downsample_factor": args.downsample_factor,
            "n_angular_bins": args.n_angular_bins,
            "angular_dr_um": args.angular_dr_um,
            "ring_radius_um": ring_r,
            "sigma_time_ring_ps": sigma_time_ring,
            "sigma_time_trigger_ps": sigma_time_trigger,
            "images": snapshot_index,
        },
    )

    n_cond = sum(1 for e in snapshot_index if e["condensed"])
    print(
        f"\n  Done: {len(selected)} images, {n_cond} condensed\n"
        f"  Snapshots: {snaps_dir}/\n"
        f"  Panel:     {panel_path}\n"
        f"  Index:     {run_dir}/snapshot_index.json"
    )


if __name__ == "__main__":
    main()
