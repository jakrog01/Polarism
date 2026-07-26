from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mnist_digits_polariton_snn_dynamic.simulation import calibration


def test_realistic_percentiles_scale_heterogeneous_powers_and_cap_candidate(
    monkeypatch,
) -> None:
    encoded_powers = np.stack(
        [
            np.array([float(value), float(value) * 0.5], dtype=np.float64)
            for value in range(1, 101)
        ]
    )
    selection = calibration._select_representative_encoded_powers(encoded_powers)
    captured_powers: list[np.ndarray] = []

    def synthetic_solver(
        resources: object,
        powers: np.ndarray,
        warmup_ps: float,
        stride_steps: int,
        snapshot: bool,
    ) -> SimpleNamespace:
        vector = np.asarray(powers, dtype=np.float64)
        captured_powers.append(vector.copy())
        return SimpleNamespace(
            psi=np.array([[1.0, 1.0], [float(np.max(vector)), 1.0]], dtype=np.float64)
        )

    monkeypatch.setattr(calibration, "simulate_one_image", synthetic_solver)
    points = calibration._run_sweep(
        SimpleNamespace(n_spots=2),
        np.array([10.0, 20.0], dtype=np.float64),
        lambda power: calibration._scaled_encoded_powers(
            selection.percentile_95_powers,
            power,
            0.0,
            100.0,
        ),
        calibration._max_spot_signal,
        0.0,
        1,
        10.0,
    )
    lower, upper = calibration._find_bracket(points, "realistic p95")
    candidate = calibration._power_max_candidate(1000.0, 100.0, 90.0, 80.0, 1.05, 0.95)

    assert selection.percentile_95_metric == np.percentile(
        np.max(encoded_powers, axis=1), 95.0
    )
    assert selection.percentile_99_metric == np.percentile(
        np.max(encoded_powers, axis=1), 99.0
    )
    assert selection.percentile_95_index == 94
    assert selection.percentile_99_index == 98
    np.testing.assert_allclose(captured_powers[0], np.array([9.5, 4.75]))
    np.testing.assert_allclose(captured_powers[1], np.array([19.0, 9.5]))
    assert lower.power == 10.0
    assert upper.power == 20.0
    assert candidate == 76.0
