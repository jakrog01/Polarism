"""Ablation readout for saved dynamic SNN traces."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from mnist_digits_polariton_snn_dynamic.config.loader import load_snn_dynamic_config
from mnist_digits_polariton_snn_dynamic.encoding.downsample import load_and_downsample
from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json
from mnist_digits_polariton_snn_dynamic.readout.classifier import train_and_evaluate
from mnist_digits_polariton_snn_dynamic.readout.features import (
    FeatureBundle,
    _build_feature_vector,
    _summarize_traces,
)


STAT_NAMES = ("peak", "t_peak", "area", "tau")


@dataclass(frozen=True, slots=True)
class AblationSubsetResult:
    """Classifier result for one ablated feature subset."""

    subset: str
    accuracy_train: float
    accuracy_test: float
    delta_vs_pixel_baseline: float
    n_features: int
    n_train: int
    n_test: int


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Scenario-level ablation report."""

    scenario_id: str
    output_dir: str
    pixel_baseline_accuracy_test: float
    subsets: list[AblationSubsetResult]
    calibration_caveat: str | None = None


def finalize_ablation(
    scenario_id: str,
    config_path: str,
    output_dir: str,
    calibration_caveat: str | None = None,
) -> AblationReport:
    """Compute pixel baseline and trace-feature ablations for one scenario."""
    out_dir = Path(output_dir).expanduser().resolve()
    labels = np.asarray(np.load(out_dir / "labels.npy"), dtype=np.int64)
    images = _load_input_images(out_dir, config_path)
    traces = _load_traces(out_dir)
    times_ps = np.asarray(np.load(out_dir / "trace_times_ps.npy"), dtype=np.float64)
    snn_cfg = load_snn_dynamic_config(config_path)
    pixel_features = np.ascontiguousarray(
        images.reshape(images.shape[0], -1),
        dtype=np.float64,
    )
    pixel_report = _evaluate_matrix(
        pixel_features,
        labels,
        snn_cfg.data.seed,
        snn_cfg.classifier.test_fraction,
        snn_cfg.classifier.C,
        snn_cfg.classifier.max_iter,
    )
    pixel_accuracy = float(pixel_report["accuracy_test"])
    subset_results = [
        _subset_result(
            "pixel_baseline",
            pixel_report,
            pixel_accuracy,
            pixel_features.shape[1],
        )
    ]
    matrices = build_ablation_matrices(traces, times_ps)
    for subset, matrix in matrices.items():
        report = _evaluate_matrix(
            matrix,
            labels,
            snn_cfg.data.seed,
            snn_cfg.classifier.test_fraction,
            snn_cfg.classifier.C,
            snn_cfg.classifier.max_iter,
        )
        subset_results.append(
            _subset_result(subset, report, pixel_accuracy, matrix.shape[1])
        )
    ablation_report = AblationReport(
        scenario_id=str(scenario_id),
        output_dir=str(out_dir),
        pixel_baseline_accuracy_test=pixel_accuracy,
        subsets=subset_results,
        calibration_caveat=calibration_caveat,
    )
    atomic_write_json(str(out_dir / "ablation_report.json"), asdict(ablation_report))
    return ablation_report


def build_ablation_matrices(
    traces: dict[str, np.ndarray],
    times_ps: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build full, field-only, stat-only, and field-stat feature matrices."""
    field_arrays = {
        name: np.asarray(array, dtype=np.float64)
        for name, array in traces.items()
    }
    n_samples = _n_samples(field_arrays)
    summaries = {
        name: np.stack(
            [
                _summarize_traces(field_arrays[name][index], times_ps)
                for index in range(n_samples)
            ],
            axis=0,
        )
        for name in field_arrays
    }
    matrices: dict[str, np.ndarray] = {
        "full_features": _full_features(field_arrays, times_ps)
    }
    for field_name, summary in summaries.items():
        matrices[f"{field_name}_all_stats"] = _flatten(summary)
        for stat_index, stat_name in enumerate(STAT_NAMES):
            matrices[f"{field_name}_{stat_name}_only"] = _flatten(
                summary[:, :, stat_index]
            )
    for stat_index, stat_name in enumerate(STAT_NAMES):
        stat_parts = [summary[:, :, stat_index] for summary in summaries.values()]
        matrices[f"{stat_name}_all_fields"] = _flatten(
            np.concatenate(stat_parts, axis=1)
        )
    return {
        key: np.ascontiguousarray(value, dtype=np.float64)
        for key, value in matrices.items()
    }


def aggregate_campaign_ablation(campaign_output_dir: str) -> dict[str, Any]:
    """Collect scenario ablation reports into `campaign_ablation_summary.json`."""
    campaign_dir = Path(campaign_output_dir).expanduser().resolve()
    reports = []
    for path in sorted(campaign_dir.glob("*/ablation_report.json")):
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        reports.append(report)
    rows = []
    for report in reports:
        for subset in report["subsets"]:
            rows.append(
                {
                    "scenario_id": report["scenario_id"],
                    "subset": subset["subset"],
                    "accuracy_test": subset["accuracy_test"],
                    "delta_vs_pixel_baseline": subset["delta_vs_pixel_baseline"],
                    "n_features": subset["n_features"],
                }
            )
    summary = {
        "campaign_output_dir": str(campaign_dir),
        "n_scenarios_with_ablation": len(reports),
        "rows": rows,
    }
    atomic_write_json(str(campaign_dir / "campaign_ablation_summary.json"), summary)
    return summary


def _load_input_images(out_dir: Path, config_path: str) -> np.ndarray:
    image_path = out_dir / "input_images.npy"
    if image_path.exists():
        return np.asarray(np.load(image_path), dtype=np.float64)
    snn_cfg = load_snn_dynamic_config(config_path)
    images, _ = load_and_downsample(
        snn_cfg.data.path,
        snn_cfg.data.n_samples,
        snn_cfg.data.seed,
        target_side=snn_cfg.encoding.n_side,
    )
    return np.asarray(images, dtype=np.float64)


def _load_traces(out_dir: Path) -> dict[str, np.ndarray]:
    traces = {"psi": np.asarray(np.load(out_dir / "traces_psi.npy"), dtype=np.float64)}
    active_path = out_dir / "traces_nA.npy"
    inactive_path = out_dir / "traces_nI.npy"
    if active_path.exists():
        traces["n_active"] = np.asarray(np.load(active_path), dtype=np.float64)
    if inactive_path.exists():
        traces["n_inactive"] = np.asarray(np.load(inactive_path), dtype=np.float64)
    return traces


def _full_features(
    traces: dict[str, np.ndarray],
    times_ps: np.ndarray,
) -> np.ndarray:
    psi = traces["psi"]
    n_active = traces.get("n_active")
    n_inactive = traces.get("n_inactive")
    rows = [
        _build_feature_vector(
            psi[index],
            n_active[index] if n_active is not None else None,
            n_inactive[index] if n_inactive is not None else None,
            times_ps,
            "both",
        )
        for index in range(psi.shape[0])
    ]
    return np.stack(rows, axis=0)


def _evaluate_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_fraction: float,
    C: float,
    max_iter: int,
) -> dict[str, Any]:
    bundle = FeatureBundle(
        features=np.ascontiguousarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        traces_psi=np.empty((labels.shape[0], 0, 0), dtype=np.float64),
        traces_n_active=None,
        traces_n_inactive=None,
        trace_times_ps=np.empty(0, dtype=np.float64),
        feature_shape=(features.shape[1],),
    )
    return asdict(
        train_and_evaluate(
            bundle,
            seed=seed,
            test_fraction=test_fraction,
            C=C,
            max_iter=max_iter,
        )
    )


def _subset_result(
    subset: str,
    report: dict[str, Any],
    pixel_accuracy: float,
    n_features: int,
) -> AblationSubsetResult:
    accuracy_test = float(report["accuracy_test"])
    return AblationSubsetResult(
        subset=str(subset),
        accuracy_train=float(report["accuracy_train"]),
        accuracy_test=accuracy_test,
        delta_vs_pixel_baseline=accuracy_test - float(pixel_accuracy),
        n_features=int(n_features),
        n_train=int(report["n_train"]),
        n_test=int(report["n_test"]),
    )


def _n_samples(traces: dict[str, np.ndarray]) -> int:
    counts = {array.shape[0] for array in traces.values()}
    if len(counts) != 1:
        raise ValueError(f"Trace arrays disagree on sample count: {sorted(counts)}")
    return int(next(iter(counts)))


def _flatten(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    return x.reshape(x.shape[0], -1)
