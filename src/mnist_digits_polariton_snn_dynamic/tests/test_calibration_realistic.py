from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mnist_digits_polariton_snn_dynamic.simulation import calibration


def test_realistic_p99_limits_candidate() -> None:
    candidate = calibration._power_max_candidate(
        1000.0,
        100.0,
        90.0,
        80.0,
        1.05,
        0.95,
    )

    assert candidate == 76.0


def test_full_lattice_margin_limits_candidate_when_realistic_is_higher() -> None:
    candidate = calibration._power_max_candidate(
        1000.0,
        100.0,
        200.0,
        300.0,
        1.05,
        0.95,
    )

    assert candidate == 105.0


def test_candidate_below_single_threshold_skips_guard(monkeypatch) -> None:
    calls: list[float] = []

    def evaluate(
        resources: object,
        powers: np.ndarray,
        scalar_power: float,
        signal_selector: object,
        warmup_ps: float,
        stride_steps: int,
        condensation_ratio: float,
    ) -> calibration.CalibrationPoint:
        calls.append(scalar_power)
        return calibration.CalibrationPoint(scalar_power, 1.0, 1.0, 1.0, False)

    monkeypatch.setattr(calibration, "_evaluate_power", evaluate)
    final_power, capped, guard_points = calibration._select_final_power(
        SimpleNamespace(n_spots=4),
        1,
        50.0,
        60.0,
        0.0,
        0.0,
        1,
        10.0,
        0.95,
        10,
    )

    assert final_power == 50.0
    assert not capped
    assert guard_points == []
    assert calls == [50.0]


def test_realistic_point_identifies_first_condensing_spot(monkeypatch) -> None:
    def synthetic_solver(
        resources: object,
        powers: np.ndarray,
        warmup_ps: float,
        stride_steps: int,
        snapshot: bool,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            psi=np.array(
                [[1.0, 1.0, 1.0, 1.0], [2.0, 12.0, 3.0, 1.0]],
                dtype=np.float64,
            )
        )

    monkeypatch.setattr(calibration, "simulate_one_image", synthetic_solver)
    point = calibration._evaluate_realistic_power(
        SimpleNamespace(),
        np.array([2.0, 5.0, 7.0], dtype=np.float64),
        5.0,
        0.0,
        1,
        10.0,
        (
            np.array([1], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ),
    )

    assert point.condensed
    assert point.first_condensing_spot_index == 1
    assert point.first_condensing_spot_local_power == 5.0
    assert point.first_condensing_spot_neighbor_mean_power == 4.5


def test_cluster_metric_prefers_neighboring_power_over_isolated_hotspot() -> None:
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    powers = np.array(
        [[100.0, 0.0, 0.0, 0.0], [40.0, 40.0, 40.0, 40.0]],
        dtype=np.float64,
    )
    metrics = calibration._direct_neighbor_cluster_metrics(
        powers,
        calibration._nearest_neighbor_indices(positions),
        3,
    )

    np.testing.assert_allclose(metrics, np.array([100.0, 120.0]))


def test_worst_threshold_index_selects_lowest_threshold_from_top_k() -> None:
    thresholds = np.array([180.0, 95.0, 120.0, 140.0], dtype=np.float64)

    assert calibration._worst_threshold_index(thresholds) == 1


def test_scaled_encoded_powers_rejects_power_below_power_min() -> None:
    with pytest.raises(ValueError, match="at least power_min"):
        calibration._scaled_encoded_powers(
            np.array([2.0, 5.0], dtype=np.float64),
            1.0,
            2.0,
            10.0,
        )
