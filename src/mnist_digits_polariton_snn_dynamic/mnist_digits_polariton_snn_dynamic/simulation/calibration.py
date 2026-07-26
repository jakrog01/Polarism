"""Condensation-threshold calibration for dynamic SNN scenarios."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from mnist_digits_polariton_snn_dynamic.config.loader import (
    build_encoder,
    build_geometry,
    load_polarism_config,
    load_snn_dynamic_config,
)
from mnist_digits_polariton_snn_dynamic.encoding.downsample import load_and_downsample
from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json
from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources
from mnist_digits_polariton_snn_dynamic.simulation.runner import simulate_one_image


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """Single power point in a condensation-threshold sweep."""

    power: float
    peak_psi_signal: float
    background_psi_signal: float
    ratio_to_background: float
    condensed: bool


@dataclass(frozen=True, slots=True)
class RealisticCalibrationPoint:
    """Single realistic-pattern power point with mixing diagnostics."""

    power: float
    peak_psi_signal: float
    background_psi_signal: float
    ratio_to_background: float
    condensed: bool
    first_condensing_spot_index: int | None
    first_condensing_spot_local_power: float | None
    first_condensing_spot_neighbor_mean_power: float | None


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationResult:
    """Threshold calibration report for synthetic and encoded pump patterns."""

    scenario_id: str
    config_path: str
    sigma_space_um: float
    n_spots: int
    center_channel: int
    power_min: float
    power_max: float
    n_points: int
    condensation_ratio: float
    full_lattice_signal: str
    single_spot_signal: str
    realistic_signal: str
    realistic_metric: str
    realistic_cluster_size: int
    realistic_top_k: int
    realistic_n_points: int
    realistic_percentile_95_metric: float
    realistic_percentile_99_metric: float
    realistic_percentile_95_image_index: int
    realistic_percentile_99_image_index: int
    realistic_percentile_95_selected_metric: float
    realistic_percentile_99_selected_metric: float
    realistic_percentile_95_candidate_indices: list[int]
    realistic_percentile_99_candidate_indices: list[int]
    realistic_percentile_95_candidate_metrics: list[float]
    realistic_percentile_99_candidate_metrics: list[float]
    realistic_percentile_95_candidate_thresholds: list[float]
    realistic_percentile_99_candidate_thresholds: list[float]
    threshold_power_full_lattice: float
    threshold_power_single_spot: float | None
    threshold_power_realistic_p95: float
    threshold_power_realistic_p99: float
    power_max_candidate: float
    final_power_max: float
    power_max_was_capped: bool
    scenario_skipped: bool
    skip_reason: str | None
    lower_power_full_lattice: float
    upper_power_full_lattice: float
    lower_power_single_spot: float | None
    upper_power_single_spot: float | None
    lower_power_realistic_p95: float
    upper_power_realistic_p95: float
    lower_power_realistic_p99: float
    upper_power_realistic_p99: float
    full_lattice_points: list[CalibrationPoint]
    single_spot_points: list[CalibrationPoint]
    realistic_p95_points: list[list[RealisticCalibrationPoint]]
    realistic_p99_points: list[list[RealisticCalibrationPoint]]
    guard_points: list[CalibrationPoint]


@dataclass(frozen=True, slots=True)
class RepresentativeEncodedPowers:
    """Representative encoded-image selections for realistic calibration."""

    percentile_95_metric: float
    percentile_99_metric: float
    percentile_95_index: int
    percentile_99_index: int
    percentile_95_selected_metric: float
    percentile_99_selected_metric: float
    percentile_95_powers: np.ndarray
    percentile_99_powers: np.ndarray
    percentile_95_candidate_indices: np.ndarray
    percentile_99_candidate_indices: np.ndarray
    percentile_95_candidate_metrics: np.ndarray
    percentile_99_candidate_metrics: np.ndarray
    percentile_95_candidate_powers: np.ndarray
    percentile_99_candidate_powers: np.ndarray


def calibrate_threshold(
    config_path: str,
    scenario_id: str,
    output_dir: str,
    *,
    n_points: int = 17,
    condensation_ratio: float = 10.0,
    full_lattice_power_margin: float = 1.05,
    single_spot_safety_fraction: float = 0.95,
    guard_n_points: int = 10,
    realistic_safety_fraction: float = 0.95,
    realistic_n_points: int = 9,
    realistic_top_k: int = 5,
    realistic_cluster_size: int = 5,
) -> ThresholdCalibrationResult:
    """Run threshold calibration and write `calibration.json`.

    Parameters
    ----------
    config_path
        Scenario YAML path.
    scenario_id
        Stable scenario identifier stored in the calibration report.
    output_dir
        Scenario output directory.
    n_points
        Number of linearly spaced powers between encoder bounds.
    condensation_ratio
        A point is condensed when peak signal exceeds this multiple of the
        first recorded signal value.
    full_lattice_power_margin
        Candidate power multiplier applied to the full-lattice threshold.
    single_spot_safety_fraction
        Maximum accepted fraction of the single-spot threshold.
    guard_n_points
        Number of descending guard powers tested when capping is needed.
    realistic_safety_fraction
        Maximum accepted fraction of each representative encoded-image threshold.
    realistic_n_points
        Number of sweep points used for each realistic encoded-image candidate.
    realistic_top_k
        Number of encoded images nearest each realistic percentile to simulate.
    realistic_cluster_size
        Number of spots in the directly neighboring lattice cluster metric.

    Returns
    -------
    ThresholdCalibrationResult
        Thresholds, final safe power, and diagnostic sweep points.
    """
    _validate_inputs(
        n_points,
        condensation_ratio,
        full_lattice_power_margin,
        single_spot_safety_fraction,
        guard_n_points,
        realistic_safety_fraction,
        realistic_n_points,
        realistic_top_k,
        realistic_cluster_size,
    )
    snn_cfg = load_snn_dynamic_config(config_path)
    polarism_cfg = load_polarism_config(snn_cfg.polarism_config_path)
    geometry = build_geometry(snn_cfg.geometry)
    encoder = build_encoder(snn_cfg.encoding)
    if encoder.n_spots != geometry.n_spots:
        raise ValueError(
            f"encoder.n_spots={encoder.n_spots} does not match "
            f"geometry.n_spots={geometry.n_spots}"
        )
    images, _ = load_and_downsample(
        snn_cfg.data.path,
        snn_cfg.data.n_samples,
        snn_cfg.data.seed,
        target_side=encoder.input_shape()[0],
    )
    encoded_powers = np.asarray(encoder.encode(images), dtype=np.float64)
    resources = SharedSimResources(
        polarism_cfg,
        geometry,
        snn_cfg.pulse,
        mask_radius_um=snn_cfg.readout.mask_radius_um,
    )
    neighbor_indices = _nearest_neighbor_indices(resources.positions)
    realistic_selection = _select_representative_encoded_powers(
        encoded_powers,
        resources.positions,
        realistic_cluster_size,
        realistic_top_k,
    )
    sweep_powers = np.linspace(
        float(snn_cfg.encoding.power_min),
        float(snn_cfg.encoding.power_max),
        int(n_points),
        dtype=np.float64,
    )
    center_channel = _center_channel(resources.positions)
    full_lattice_points = _run_sweep(
        resources,
        sweep_powers,
        lambda power: np.full(resources.n_spots, power, dtype=np.float64),
        _max_spot_signal,
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
    )
    full_lower, full_upper = _find_bracket(
        full_lattice_points, "full-lattice", fallback_to_top=True
    )
    threshold_full = 0.5 * (float(full_lower.power) + float(full_upper.power))
    single_spot_points = _run_sweep(
        resources,
        sweep_powers,
        lambda power: _single_spot_powers(resources.n_spots, center_channel, power),
        lambda traces: np.asarray(traces[:, center_channel], dtype=np.float64),
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
    )
    single_bracket = _find_optional_bracket(single_spot_points)
    if single_bracket is None:
        threshold_single = None
        single_lower = None
        single_upper = None
    else:
        single_lower, single_upper = single_bracket
        threshold_single = 0.5 * (float(single_lower.power) + float(single_upper.power))
    realistic_sweep_powers = np.linspace(
        float(snn_cfg.encoding.power_min),
        float(snn_cfg.encoding.power_max),
        int(realistic_n_points),
        dtype=np.float64,
    )
    realistic_p95_points, realistic_p95_thresholds = _run_realistic_sweeps(
        resources,
        realistic_sweep_powers,
        realistic_selection.percentile_95_candidate_powers,
        float(snn_cfg.encoding.power_min),
        float(snn_cfg.encoding.power_max),
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
        neighbor_indices,
        "realistic p95",
    )
    realistic_p95_worst = _worst_threshold_index(realistic_p95_thresholds)
    realistic_p95_lower, realistic_p95_upper = _find_bracket(
        realistic_p95_points[realistic_p95_worst],
        "realistic p95",
        fallback_to_top=True,
    )
    threshold_realistic_p95 = float(realistic_p95_thresholds[realistic_p95_worst])
    realistic_p99_points, realistic_p99_thresholds = _run_realistic_sweeps(
        resources,
        realistic_sweep_powers,
        realistic_selection.percentile_99_candidate_powers,
        float(snn_cfg.encoding.power_min),
        float(snn_cfg.encoding.power_max),
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
        neighbor_indices,
        "realistic p99",
    )
    realistic_p99_worst = _worst_threshold_index(realistic_p99_thresholds)
    realistic_p99_lower, realistic_p99_upper = _find_bracket(
        realistic_p99_points[realistic_p99_worst],
        "realistic p99",
        fallback_to_top=True,
    )
    threshold_realistic_p99 = float(realistic_p99_thresholds[realistic_p99_worst])
    candidate = _power_max_candidate(
        float(snn_cfg.encoding.power_max),
        float(threshold_full),
        float(threshold_realistic_p95),
        float(threshold_realistic_p99),
        float(full_lattice_power_margin),
        float(realistic_safety_fraction),
    )
    final_power, capped, guard_points = _select_final_power(
        resources,
        center_channel,
        candidate,
        threshold_single,
        float(snn_cfg.encoding.power_min),
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
        float(single_spot_safety_fraction),
        int(guard_n_points),
    )
    final_full_point = _evaluate_power(
        resources,
        _full_lattice_powers(resources.n_spots, final_power),
        final_power,
        _max_spot_signal,
        snn_cfg.readout.warmup_ps,
        snn_cfg.readout.stride_steps,
        float(condensation_ratio),
    )
    scenario_skipped = not final_full_point.condensed
    skip_reason: str | None = None
    if scenario_skipped:
        skip_reason = (
            f"final calibrated power_max={final_power} does not condense the full "
            f"lattice (ratio_to_background={final_full_point.ratio_to_background:.3f} "
            f"vs required {float(condensation_ratio)}); downstream run/finalize will "
            f"be skipped. Increase encoding.power_max, lower condensation_ratio, or "
            f"revise scenario geometry/pulse settings."
        )
        print(f"WARNING: {skip_reason}", file=sys.stderr, flush=True)
    result = ThresholdCalibrationResult(
        scenario_id=str(scenario_id),
        config_path=str(Path(config_path).expanduser().resolve()),
        sigma_space_um=float(snn_cfg.geometry.sigma_space_um),
        n_spots=int(resources.n_spots),
        center_channel=int(center_channel),
        power_min=float(snn_cfg.encoding.power_min),
        power_max=float(snn_cfg.encoding.power_max),
        n_points=int(n_points),
        condensation_ratio=float(condensation_ratio),
        full_lattice_signal="max_per_spot_channel",
        single_spot_signal=f"center_spot_channel_{center_channel}",
        realistic_signal="max_per_spot_channel",
        realistic_metric="max_direct_neighbor_cluster_power",
        realistic_cluster_size=int(realistic_cluster_size),
        realistic_top_k=int(realistic_top_k),
        realistic_n_points=int(realistic_n_points),
        realistic_percentile_95_metric=float(realistic_selection.percentile_95_metric),
        realistic_percentile_99_metric=float(realistic_selection.percentile_99_metric),
        realistic_percentile_95_image_index=int(
            realistic_selection.percentile_95_candidate_indices[realistic_p95_worst]
        ),
        realistic_percentile_99_image_index=int(
            realistic_selection.percentile_99_candidate_indices[realistic_p99_worst]
        ),
        realistic_percentile_95_selected_metric=float(
            realistic_selection.percentile_95_candidate_metrics[realistic_p95_worst]
        ),
        realistic_percentile_99_selected_metric=float(
            realistic_selection.percentile_99_candidate_metrics[realistic_p99_worst]
        ),
        realistic_percentile_95_candidate_indices=[
            int(index) for index in realistic_selection.percentile_95_candidate_indices
        ],
        realistic_percentile_99_candidate_indices=[
            int(index) for index in realistic_selection.percentile_99_candidate_indices
        ],
        realistic_percentile_95_candidate_metrics=[
            float(metric) for metric in realistic_selection.percentile_95_candidate_metrics
        ],
        realistic_percentile_99_candidate_metrics=[
            float(metric) for metric in realistic_selection.percentile_99_candidate_metrics
        ],
        realistic_percentile_95_candidate_thresholds=[
            float(threshold) for threshold in realistic_p95_thresholds
        ],
        realistic_percentile_99_candidate_thresholds=[
            float(threshold) for threshold in realistic_p99_thresholds
        ],
        threshold_power_full_lattice=float(threshold_full),
        threshold_power_single_spot=(
            float(threshold_single) if threshold_single is not None else None
        ),
        threshold_power_realistic_p95=float(threshold_realistic_p95),
        threshold_power_realistic_p99=float(threshold_realistic_p99),
        power_max_candidate=float(candidate),
        final_power_max=float(final_power),
        power_max_was_capped=bool(capped),
        scenario_skipped=bool(scenario_skipped),
        skip_reason=skip_reason,
        lower_power_full_lattice=float(full_lower.power),
        upper_power_full_lattice=float(full_upper.power),
        lower_power_single_spot=(
            float(single_lower.power) if single_lower is not None else None
        ),
        upper_power_single_spot=(
            float(single_upper.power) if single_upper is not None else None
        ),
        lower_power_realistic_p95=float(realistic_p95_lower.power),
        upper_power_realistic_p95=float(realistic_p95_upper.power),
        lower_power_realistic_p99=float(realistic_p99_lower.power),
        upper_power_realistic_p99=float(realistic_p99_upper.power),
        full_lattice_points=full_lattice_points,
        single_spot_points=single_spot_points,
        realistic_p95_points=realistic_p95_points,
        realistic_p99_points=realistic_p99_points,
        guard_points=guard_points,
    )
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(out_dir / "calibration.json"), asdict(result))
    return result


def final_power_max(calibration_path: str) -> float:
    """Return the final calibrated power_max from `calibration.json`."""
    calibration = _load_calibration(calibration_path)
    if "final_power_max" not in calibration:
        raise ValueError(f"Calibration is missing final_power_max: {calibration_path}")
    power = float(calibration["final_power_max"])
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError(f"Calibration final_power_max must be positive, got {power}")
    return power


def is_scenario_skipped(calibration_path: str) -> tuple[bool, str | None]:
    """Return whether the scenario should be skipped and the recorded reason."""
    calibration = _load_calibration(calibration_path)
    skipped = bool(calibration.get("scenario_skipped", False))
    reason = calibration.get("skip_reason")
    return skipped, (str(reason) if reason is not None else None)


def main() -> None:
    """Run threshold calibration from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-points", type=int, default=17)
    parser.add_argument("--condensation-ratio", type=float, default=10.0)
    parser.add_argument("--full-lattice-power-margin", type=float, default=1.05)
    parser.add_argument("--single-spot-safety-fraction", type=float, default=0.95)
    parser.add_argument("--realistic-safety-fraction", type=float, default=0.95)
    parser.add_argument("--guard-n-points", type=int, default=10)
    args = parser.parse_args()
    result = calibrate_threshold(
        args.config,
        args.scenario_id,
        args.output_dir,
        n_points=args.n_points,
        condensation_ratio=args.condensation_ratio,
        full_lattice_power_margin=args.full_lattice_power_margin,
        single_spot_safety_fraction=args.single_spot_safety_fraction,
        guard_n_points=args.guard_n_points,
        realistic_safety_fraction=args.realistic_safety_fraction,
    )
    print(json.dumps(asdict(result), indent=2))


def _validate_inputs(
    n_points: int,
    condensation_ratio: float,
    full_lattice_power_margin: float,
    single_spot_safety_fraction: float,
    guard_n_points: int,
    realistic_safety_fraction: float,
    realistic_n_points: int,
    realistic_top_k: int,
    realistic_cluster_size: int,
) -> None:
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    if condensation_ratio <= 1.0:
        raise ValueError(
            f"condensation_ratio must be greater than 1, got {condensation_ratio}"
        )
    if full_lattice_power_margin <= 1.0:
        raise ValueError(
            "full_lattice_power_margin must be greater than 1, "
            f"got {full_lattice_power_margin}"
        )
    if not 0.0 < single_spot_safety_fraction < 1.0:
        raise ValueError(
            "single_spot_safety_fraction must be in (0, 1), "
            f"got {single_spot_safety_fraction}"
        )
    if not 0.0 < realistic_safety_fraction < 1.0:
        raise ValueError(
            "realistic_safety_fraction must be in (0, 1), "
            f"got {realistic_safety_fraction}"
        )
    if guard_n_points < 2:
        raise ValueError(f"guard_n_points must be at least 2, got {guard_n_points}")
    if realistic_n_points < 2:
        raise ValueError(
            f"realistic_n_points must be at least 2, got {realistic_n_points}"
        )
    if realistic_top_k < 1:
        raise ValueError(f"realistic_top_k must be at least 1, got {realistic_top_k}")
    if realistic_cluster_size < 2:
        raise ValueError(
            "realistic_cluster_size must be at least 2, "
            f"got {realistic_cluster_size}"
        )


def _run_sweep(
    resources: SharedSimResources,
    powers: np.ndarray,
    power_vector_factory: Callable[[float], np.ndarray],
    signal_selector: Callable[[np.ndarray], np.ndarray],
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
) -> list[CalibrationPoint]:
    return [
        _evaluate_power(
            resources,
            power_vector_factory(float(power)),
            float(power),
            signal_selector,
            warmup_ps,
            stride_steps,
            condensation_ratio,
        )
        for power in powers
    ]


def _run_realistic_sweeps(
    resources: SharedSimResources,
    sweep_powers: np.ndarray,
    encoded_power_candidates: np.ndarray,
    power_min: float,
    power_max: float,
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
    neighbor_indices: tuple[np.ndarray, ...],
    name: str,
) -> tuple[list[list[RealisticCalibrationPoint]], np.ndarray]:
    sweeps = [
        _run_realistic_sweep(
            resources,
            sweep_powers,
            encoded_powers,
            power_min,
            power_max,
            warmup_ps,
            stride_steps,
            condensation_ratio,
            neighbor_indices,
        )
        for encoded_powers in np.asarray(encoded_power_candidates, dtype=np.float64)
    ]
    thresholds = np.asarray(
        [_threshold_from_points(points, name) for points in sweeps],
        dtype=np.float64,
    )
    return sweeps, thresholds


def _run_realistic_sweep(
    resources: SharedSimResources,
    sweep_powers: np.ndarray,
    encoded_powers: np.ndarray,
    power_min: float,
    power_max: float,
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
    neighbor_indices: tuple[np.ndarray, ...],
) -> list[RealisticCalibrationPoint]:
    return [
        _evaluate_realistic_power(
            resources,
            _scaled_encoded_powers(encoded_powers, float(power), power_min, power_max),
            float(power),
            warmup_ps,
            stride_steps,
            condensation_ratio,
            neighbor_indices,
        )
        for power in sweep_powers
    ]


def _select_representative_encoded_powers(
    encoded_powers: np.ndarray,
    positions: np.ndarray | None = None,
    cluster_size: int = 1,
    top_k: int = 1,
) -> RepresentativeEncodedPowers:
    powers = np.asarray(encoded_powers, dtype=np.float64)
    if powers.ndim != 2 or powers.shape[0] == 0 or powers.shape[1] == 0:
        raise ValueError(
            "encoded_powers must have non-zero shape (n_images, n_spots), "
            f"got {powers.shape}"
        )
    if not np.all(np.isfinite(powers)):
        raise ValueError("encoded_powers must contain only finite values")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if top_k > powers.shape[0]:
        raise ValueError(
            f"top_k={top_k} exceeds the number of encoded images={powers.shape[0]}"
        )
    if positions is None:
        metrics = np.max(powers, axis=1)
    else:
        # WHY: a bright connected neighborhood captures inter-spot amplification absent from isolated hotspots.
        metrics = _direct_neighbor_cluster_metrics(
            powers,
            _nearest_neighbor_indices(positions),
            cluster_size,
        )
    percentile_95_metric = float(np.percentile(metrics, 95.0))
    percentile_99_metric = float(np.percentile(metrics, 99.0))
    percentile_95_candidate_indices = _nearest_metric_indices(
        metrics,
        percentile_95_metric,
        top_k,
    )
    percentile_99_candidate_indices = _nearest_metric_indices(
        metrics,
        percentile_99_metric,
        top_k,
    )
    percentile_95_index = int(percentile_95_candidate_indices[0])
    percentile_99_index = int(percentile_99_candidate_indices[0])
    return RepresentativeEncodedPowers(
        percentile_95_metric=percentile_95_metric,
        percentile_99_metric=percentile_99_metric,
        percentile_95_index=percentile_95_index,
        percentile_99_index=percentile_99_index,
        percentile_95_selected_metric=float(metrics[percentile_95_index]),
        percentile_99_selected_metric=float(metrics[percentile_99_index]),
        percentile_95_powers=np.ascontiguousarray(
            powers[percentile_95_index],
            dtype=np.float64,
        ),
        percentile_99_powers=np.ascontiguousarray(
            powers[percentile_99_index],
            dtype=np.float64,
        ),
        percentile_95_candidate_indices=percentile_95_candidate_indices,
        percentile_99_candidate_indices=percentile_99_candidate_indices,
        percentile_95_candidate_metrics=np.ascontiguousarray(
            metrics[percentile_95_candidate_indices],
            dtype=np.float64,
        ),
        percentile_99_candidate_metrics=np.ascontiguousarray(
            metrics[percentile_99_candidate_indices],
            dtype=np.float64,
        ),
        percentile_95_candidate_powers=np.ascontiguousarray(
            powers[percentile_95_candidate_indices],
            dtype=np.float64,
        ),
        percentile_99_candidate_powers=np.ascontiguousarray(
            powers[percentile_99_candidate_indices],
            dtype=np.float64,
        ),
    )


def _nearest_metric_index(metrics: np.ndarray, percentile_metric: float) -> int:
    return int(np.argmin(np.abs(np.asarray(metrics, dtype=np.float64) - percentile_metric)))


def _nearest_metric_indices(
    metrics: np.ndarray,
    percentile_metric: float,
    top_k: int,
) -> np.ndarray:
    values = np.asarray(metrics, dtype=np.float64)
    indices = np.arange(values.size, dtype=np.int64)
    order = np.lexsort((indices, np.abs(values - float(percentile_metric))))
    return np.ascontiguousarray(order[: int(top_k)], dtype=np.int64)


def _worst_threshold_index(thresholds: np.ndarray) -> int:
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"thresholds must be a non-empty vector, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("thresholds must contain only finite values")
    return int(np.argmin(values))


def _nearest_neighbor_indices(positions: np.ndarray) -> tuple[np.ndarray, ...]:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] == 0:
        raise ValueError(f"positions must have shape (n_spots, 2), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")
    if pos.shape[0] == 1:
        return (np.empty(0, dtype=np.int64),)
    deltas = pos[:, None, :] - pos[None, :, :]
    squared_distances = np.sum(deltas * deltas, axis=2, dtype=np.float64)
    positive_distances = squared_distances[squared_distances > 0.0]
    if positive_distances.size == 0:
        raise ValueError("positions must not contain duplicate spot coordinates")
    nearest_distance = float(np.min(positive_distances))
    tolerance = np.finfo(np.float64).eps * max(1.0, nearest_distance) * 32.0
    adjacency = np.abs(squared_distances - nearest_distance) <= tolerance
    np.fill_diagonal(adjacency, False)
    return tuple(
        np.flatnonzero(adjacency[index]).astype(np.int64, copy=False)
        for index in range(pos.shape[0])
    )


def _direct_neighbor_cluster_metrics(
    encoded_powers: np.ndarray,
    neighbor_indices: tuple[np.ndarray, ...],
    cluster_size: int,
) -> np.ndarray:
    powers = np.asarray(encoded_powers, dtype=np.float64)
    if powers.ndim != 2 or powers.shape[1] != len(neighbor_indices):
        raise ValueError(
            "encoded_powers and neighbor_indices dimensions disagree, "
            f"got {powers.shape} and {len(neighbor_indices)}"
        )
    if cluster_size < 2:
        raise ValueError(f"cluster_size must be at least 2, got {cluster_size}")
    neighborhood_sums = np.empty_like(powers, dtype=np.float64)
    for spot_index, neighbors in enumerate(neighbor_indices):
        local = powers[:, spot_index]
        neighbor_powers = powers[:, neighbors]
        strongest_neighbors = np.sort(neighbor_powers, axis=1)[:, -(cluster_size - 1) :]
        neighborhood_sums[:, spot_index] = local + np.sum(
            strongest_neighbors,
            axis=1,
            dtype=np.float64,
        )
    return np.max(neighborhood_sums, axis=1)


def _scaled_encoded_powers(
    encoded_powers: np.ndarray,
    power: float,
    power_min: float,
    power_max: float,
) -> np.ndarray:
    """Scale a pattern for an effective inference-time encoder power_max; at power_min its range intentionally collapses."""
    source = np.asarray(encoded_powers, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError(
            "encoded_powers must be a non-empty one-dimensional vector, "
            f"got {source.shape}"
        )
    if not power_min < power_max:
        raise ValueError(f"power_min must be below power_max, got {power_min}, {power_max}")
    if power < 0.0:
        raise ValueError(f"power must be non-negative, got {power}")
    if power < power_min:
        raise ValueError(f"power must be at least power_min={power_min}, got {power}")
    normalized = (source - float(power_min)) / (float(power_max) - float(power_min))
    return np.ascontiguousarray(
        float(power_min) + normalized * (float(power) - float(power_min)),
        dtype=np.float64,
    )


def _power_max_candidate(
    power_max: float,
    threshold_full: float,
    threshold_realistic_p95: float,
    threshold_realistic_p99: float,
    full_lattice_power_margin: float,
    realistic_safety_fraction: float,
) -> float:
    return float(
        min(
            float(power_max),
            float(threshold_full) * float(full_lattice_power_margin),
            float(threshold_realistic_p95) * float(realistic_safety_fraction),
            float(threshold_realistic_p99) * float(realistic_safety_fraction),
        )
    )


def _evaluate_power(
    resources: SharedSimResources,
    powers: np.ndarray,
    scalar_power: float,
    signal_selector: Callable[[np.ndarray], np.ndarray],
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
) -> CalibrationPoint:
    signal, peak, background, ratio, condensed = _evaluate_signal(
        resources,
        np.asarray(powers, dtype=np.float64),
        warmup_ps,
        stride_steps,
        signal_selector,
        condensation_ratio,
    )
    return CalibrationPoint(
        power=float(scalar_power),
        peak_psi_signal=peak,
        background_psi_signal=background,
        ratio_to_background=ratio,
        condensed=condensed,
    )


def _evaluate_realistic_power(
    resources: SharedSimResources,
    powers: np.ndarray,
    scalar_power: float,
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
    neighbor_indices: tuple[np.ndarray, ...],
) -> RealisticCalibrationPoint:
    vector = np.asarray(powers, dtype=np.float64)
    signal, peak, background, ratio, condensed, spots = _evaluate_signal(
        resources,
        vector,
        warmup_ps,
        stride_steps,
        _max_spot_signal,
        condensation_ratio,
        include_spots=True,
    )
    spot_index = _first_condensing_spot_index(spots, background, condensation_ratio)
    if spot_index is None:
        local_power = None
        neighbor_mean_power = None
    else:
        local_power = float(vector[spot_index])
        neighbors = neighbor_indices[spot_index]
        neighbor_mean_power = (
            float(np.mean(vector[neighbors], dtype=np.float64))
            if neighbors.size > 0
            else None
        )
    return RealisticCalibrationPoint(
        power=float(scalar_power),
        peak_psi_signal=peak,
        background_psi_signal=background,
        ratio_to_background=ratio,
        condensed=condensed,
        first_condensing_spot_index=spot_index,
        first_condensing_spot_local_power=local_power,
        first_condensing_spot_neighbor_mean_power=neighbor_mean_power,
    )


def _evaluate_signal(
    resources: SharedSimResources,
    powers: np.ndarray,
    warmup_ps: float,
    stride_steps: int,
    signal_selector: Callable[[np.ndarray], np.ndarray],
    condensation_ratio: float,
    include_spots: bool = False,
) -> tuple[np.ndarray, float, float, float, bool] | tuple[
    np.ndarray,
    float,
    float,
    float,
    bool,
    np.ndarray,
]:
    traces = simulate_one_image(
        resources,
        np.asarray(powers, dtype=np.float64),
        warmup_ps,
        stride_steps,
        False,
    )
    psi = np.asarray(traces.psi, dtype=np.float64)
    signal = np.asarray(signal_selector(psi), dtype=np.float64)
    background = float(signal[0])
    peak = float(np.max(signal))
    if background <= 0.0:
        ratio = float(np.inf if peak > 0.0 else 1.0)
    else:
        ratio = float(peak / background)
    condensed = bool(ratio > condensation_ratio)
    if include_spots:
        return signal, peak, background, ratio, condensed, np.asarray(psi[:, :-1], dtype=np.float64)
    return signal, peak, background, ratio, condensed


def _first_condensing_spot_index(
    spots: np.ndarray,
    background: float,
    condensation_ratio: float,
) -> int | None:
    values = np.asarray(spots, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"spots must have non-zero shape (n_times, n_spots), got {values.shape}")
    if background <= 0.0:
        crossings = values > 0.0
        ratios = np.where(crossings, np.inf, 1.0)
    else:
        ratios = values / float(background)
        crossings = ratios > float(condensation_ratio)
    first_times = np.flatnonzero(np.any(crossings, axis=1))
    if first_times.size == 0:
        return None
    first_time = int(first_times[0])
    return int(np.argmax(ratios[first_time]))


def _select_final_power(
    resources: SharedSimResources,
    center_channel: int,
    candidate: float,
    threshold_single: float | None,
    power_min: float,
    warmup_ps: float,
    stride_steps: int,
    condensation_ratio: float,
    safety_fraction: float,
    guard_n_points: int,
) -> tuple[float, bool, list[CalibrationPoint]]:
    candidate_point = _evaluate_power(
        resources,
        _single_spot_powers(resources.n_spots, center_channel, float(candidate)),
        float(candidate),
        lambda traces: np.asarray(traces[:, center_channel], dtype=np.float64),
        warmup_ps,
        stride_steps,
        condensation_ratio,
    )
    if not candidate_point.condensed and (
        threshold_single is None or candidate < threshold_single
    ):
        return float(candidate), False, []
    if threshold_single is None:
        raise ValueError(
            "Single-spot guard candidate condensed, but the sweep did not find "
            "a usable non-condensed to condensed bracket. Increase n_points or "
            "revise the sweep power range."
        )
    guard_powers = np.linspace(
        float(threshold_single) * float(safety_fraction),
        float(power_min),
        int(guard_n_points),
        dtype=np.float64,
    )
    guard_points = [
        _evaluate_power(
            resources,
            _single_spot_powers(resources.n_spots, center_channel, float(power)),
            float(power),
            lambda traces: np.asarray(traces[:, center_channel], dtype=np.float64),
            warmup_ps,
            stride_steps,
            condensation_ratio,
        )
        for power in guard_powers
    ]
    safe_points = [point for point in guard_points if not point.condensed]
    if not safe_points:
        raise ValueError(
            "Single-spot guard found no safe power below threshold. "
            "Reduce single_spot_safety_fraction or increase guard_n_points."
        )
    return float(safe_points[0].power), True, guard_points


def _threshold_from_points(
    points: list[CalibrationPoint] | list[RealisticCalibrationPoint],
    name: str,
) -> float:
    lower, upper = _find_bracket(points, name, fallback_to_top=True)
    return 0.5 * (float(lower.power) + float(upper.power))


def _find_bracket(
    points: list[CalibrationPoint] | list[RealisticCalibrationPoint],
    name: str,
    *,
    fallback_to_top: bool = False,
) -> tuple[CalibrationPoint | RealisticCalibrationPoint, CalibrationPoint | RealisticCalibrationPoint]:
    bracket = _find_optional_bracket(points)
    if bracket is not None:
        return bracket
    condensed = [point.condensed for point in points]
    if fallback_to_top and points:
        top = points[-1]
        print(
            f"WARNING: calibration sweep did not find a non-condensed -> condensed "
            f"{name} bracket; condensed_flags={condensed}; falling back to top of "
            f"sweep power={top.power} (threshold treated as >= power_max).",
            file=sys.stderr,
            flush=True,
        )
        return top, top
    raise ValueError(
        f"Calibration did not find a non-condensed to condensed {name} bracket; "
        f"condensed_flags={condensed}"
    )


def _find_optional_bracket(
    points: list[CalibrationPoint] | list[RealisticCalibrationPoint],
) -> (
    tuple[CalibrationPoint | RealisticCalibrationPoint, CalibrationPoint | RealisticCalibrationPoint]
    | None
):
    for lower, upper in zip(points[:-1], points[1:]):
        if not lower.condensed and upper.condensed:
            return lower, upper
    return None


def _center_channel(positions: np.ndarray) -> int:
    pos = np.asarray(positions, dtype=np.float64)
    distances = np.sum(pos * pos, axis=1, dtype=np.float64)
    return int(np.argmin(distances))


def _full_lattice_powers(n_spots: int, power: float) -> np.ndarray:
    return np.full(int(n_spots), float(power), dtype=np.float64)


def _single_spot_powers(n_spots: int, center_channel: int, power: float) -> np.ndarray:
    powers = np.zeros(int(n_spots), dtype=np.float64)
    powers[int(center_channel)] = float(power)
    return powers


def _max_spot_signal(traces_psi: np.ndarray) -> np.ndarray:
    spots = np.asarray(traces_psi[:, :-1], dtype=np.float64)
    return np.max(spots, axis=1)


def _load_calibration(path: str) -> dict[str, object]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file must contain an object: {path}")
    return data


if __name__ == "__main__":
    main()
