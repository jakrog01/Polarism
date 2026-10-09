from __future__ import annotations

import numpy as np
import pytest

from polarism.init_condition import make_initial_psi

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


def test_gaussian_initialisation_invariants_and_filter() -> None:
    eps = 1e-3
    psi = make_initial_psi(np, 64, 64, eps, mode="complex_gaussian_zero_mean", seed=3)
    assert np.isclose(np.sqrt(np.mean(np.abs(psi) ** 2)), eps, rtol=TOL_MACHINE_F64)
    assert abs(np.mean(psi)) < TOL_MACHINE_F64
    filtered = make_initial_psi(np, 64, 64, eps, mode="filtered_complex_gaussian", dx=0.5, dy=0.5, k_cutoff_um=1.0, seed=3)
    k = 2 * np.pi * np.fft.fftfreq(64, d=0.5)
    mask = np.sqrt(np.meshgrid(k, k)[0] ** 2 + np.meshgrid(k, k)[1] ** 2) > 1.0
    assert np.isclose(np.sqrt(np.mean(np.abs(filtered) ** 2)), eps, rtol=TOL_MACHINE_F64)
    assert abs(np.mean(filtered)) < TOL_MACHINE_F64
    assert np.max(np.abs(np.fft.fft2(filtered)[mask])) < TOL_MACHINE_F64


def test_legacy_and_seed_contracts() -> None:
    eps = 1e-3
    legacy = make_initial_psi(np, 256, 256, eps, seed=7)
    assert abs(legacy.real.mean() - eps / 2) < eps / 50
    assert abs(legacy.imag.mean() - eps / 2) < eps / 50
    assert abs(legacy.mean()) > eps / 10 and not np.isclose(np.sqrt(np.mean(np.abs(legacy) ** 2)), eps)
    first = make_initial_psi(np, 16, 16, eps, mode="complex_gaussian_zero_mean", seed=9)
    assert np.array_equal(first, make_initial_psi(np, 16, 16, eps, mode="complex_gaussian_zero_mean", seed=9))
    assert not np.array_equal(first, make_initial_psi(np, 16, 16, eps, mode="complex_gaussian_zero_mean", seed=10))


def test_invalid_initialisation_modes() -> None:
    with pytest.raises(ValueError, match="DC bin"):
        make_initial_psi(np, 16, 16, 1e-3, mode="filtered_complex_gaussian", dx=1.0, dy=1.0, k_cutoff_um=0.1, seed=1)
    with pytest.raises(ValueError, match="legacy_positive_uniform"):
        make_initial_psi(np, 8, 8, 1e-3, mode="bad", seed=1)
