"""Square-4 fringe analysis experiment: 4-spot square geometry + fringe summary."""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


_FRINGE_CSV_COLUMNS: list[str] = [
    "scenario", "square_side_um", "energy", "sigma_space",
    "fringe_contrast_max", "t_fringe_contrast_max_ps",
    "fringe_spacing_at_max_contrast_um", "fringe_fft_peak_k_at_max_contrast",
    "fringe_cv_max", "h_contrast_max", "v_contrast_max",
    "fringe_window_psi_sq_max", "t_fringe_window_psi_sq_max_ps",
    "central_roi_peak_psi_sq", "central_roi_peak_emission",
    "crossed_threshold",
]


def _load_fringe_json(run_dir: str, scenario_name: str) -> dict | None:
    path = os.path.join(run_dir, f"{scenario_name}_fringe.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_sidecar_roi_peaks(run_dir: str, scenario_name: str) -> dict[str, float]:
    path = os.path.join(run_dir, f"{scenario_name}_scalars.npz")
    if not os.path.isfile(path):
        return {}
    data = np.load(path)
    out: dict[str, float] = {}
    prefix = "roi_center_D_circle"
    for suffix, key in (
        ("_mean_psi_sq", "central_roi_peak_psi_sq"),
        ("_integral_emission", "central_roi_peak_emission"),
    ):
        arr_key = f"{prefix}{suffix}"
        if arr_key in data:
            out[key] = float(data[arr_key].max())
    return out


def _is_nan(v: Any) -> bool:
    try:
        return v is None or float(v) != float(v)
    except (TypeError, ValueError):
        return True


def _generate_selected_extremes(rows: list[dict], results_dir: str) -> None:
    from pipeline.manifest.io import atomic_write_json

    if not rows:
        return

    def _pick(key: str, best_fn: Any) -> dict | None:
        valid = [r for r in rows if not _is_nan(r.get(key))]
        return best_fn(valid, key=lambda r: r[key]) if valid else None

    above = [r for r in rows if r.get("crossed_threshold")]
    below = [r for r in rows if not r.get("crossed_threshold")]

    extremes: dict = {}
    r = _pick("central_roi_peak_psi_sq", max)
    if r:
        extremes["max_central_psi_sq"] = r
    r = _pick("central_roi_peak_psi_sq", min)
    if r:
        extremes["min_central_psi_sq"] = r
    r = _pick("fringe_contrast_max", max)
    if r:
        extremes["max_fringe_contrast"] = r
    if above:
        extremes["first_above_threshold"] = min(
            above, key=lambda r: r.get("energy", float("inf"))
        )
    if below:
        valid_below = [r for r in below if not _is_nan(r.get("central_roi_peak_psi_sq"))]
        if valid_below:
            extremes["closest_below_threshold"] = max(
                valid_below, key=lambda r: r["central_roi_peak_psi_sq"]
            )

    atomic_write_json(os.path.join(results_dir, "selected_extremes.json"), extremes)


def _generate_fringe_summary(
    scenarios: list[str], run_dir: str, results_dir: str
) -> None:
    import csv as _csv
    from pipeline.manifest.io import atomic_write_json, load_scenario_meta

    rows: list[dict] = []
    for name in scenarios:
        fringe = _load_fringe_json(run_dir, name)
        if fringe is None or "aggregated" not in fringe:
            continue
        meta = load_scenario_meta(run_dir, name)
        sweep = meta.get("sweep") or {}
        agg = fringe["aggregated"]
        roi = _load_sidecar_roi_peaks(run_dir, name)
        row: dict = {
            "scenario": name,
            "square_side_um": sweep.get("square_side", float("nan")),
            "energy": sweep.get("power", float("nan")),
            "sigma_space": sweep.get("sigma_space", float("nan")),
            "fringe_contrast_max": agg.get("fringe_contrast_max", float("nan")),
            "t_fringe_contrast_max_ps": agg.get("t_fringe_contrast_max", float("nan")),
            "fringe_spacing_at_max_contrast_um": agg.get("fringe_spacing_at_max_contrast", float("nan")),
            "fringe_fft_peak_k_at_max_contrast": agg.get("fringe_fft_peak_k_at_max_contrast", float("nan")),
            "fringe_cv_max": agg.get("fringe_cv_max", float("nan")),
            "h_contrast_max": agg.get("h_contrast_max", float("nan")),
            "v_contrast_max": agg.get("v_contrast_max", float("nan")),
            "fringe_window_psi_sq_max": agg.get("fringe_window_psi_sq_max", float("nan")),
            "t_fringe_window_psi_sq_max_ps": agg.get("t_fringe_window_psi_sq_max", float("nan")),
            "central_roi_peak_psi_sq": roi.get("central_roi_peak_psi_sq", float("nan")),
            "central_roi_peak_emission": roi.get("central_roi_peak_emission", float("nan")),
            "crossed_threshold": agg.get("crossed_threshold", False),
        }
        rows.append(row)

    if not rows:
        return

    csv_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=_FRINGE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.json")
    atomic_write_json(json_path, rows)

    _generate_selected_extremes(rows, results_dir)
    print(f"  Fringe summary: {csv_path}  ({len(rows)} scenarios)")


class Square4FringeExperiment:
    """Square-4 spatiotemporal fringe analysis experiment."""

    name = "square4_fringe"

    def matches(self, cfg: dict[str, Any]) -> bool:
        return any(
            sc.get("geometry") == "square4"
            for sc in cfg.get("scenarios", [])
        )

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        return []

    def expand_parameter_sweep(
        self, cfg: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
        from pipeline.config.sweep import expand_generic
        return expand_generic(cfg)

    def build_calibration_scenarios(
        self, cfg: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        return {}

    def summarize(
        self, scenarios: list[str], run_dir: str, results_dir: str
    ) -> bool:
        if not scenarios:
            return False
        _generate_fringe_summary(scenarios, run_dir, results_dir)
        return True
