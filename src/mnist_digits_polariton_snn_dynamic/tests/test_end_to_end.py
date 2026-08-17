from __future__ import annotations

from pathlib import Path

import numpy as np

from polarism.config.simulation_parameters import Config, GridParameters
from mnist_digits_polariton_snn_dynamic.config.loader import (
    EncodingConfig,
    GeometryConfig,
    PulseConfig,
    ReadoutConfig,
    build_encoder,
    build_geometry,
)
from mnist_digits_polariton_snn_dynamic.encoding.downsample import load_and_downsample
from mnist_digits_polariton_snn_dynamic.readout.classifier import (
    ClassificationReport,
    train_and_evaluate,
)
from mnist_digits_polariton_snn_dynamic.readout.features import FeatureBundle, collect_features
from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources


def _synthetic_digits(path: Path) -> None:
    rng = np.random.default_rng(20260816)
    images = []
    labels = []
    for label in range(10):
        for variant in range(2):
            image = np.zeros((7, 7), dtype=np.float64)
            image[label % 7, :] = 0.65
            image[:, (2 * label + variant) % 7] += 0.25
            image += 0.05 * rng.random((7, 7))
            images.append(np.clip(image, 0.0, 1.0))
            labels.append(label)
    pixels = np.rint(np.repeat(np.repeat(images, 4, axis=1), 4, axis=2) * 255.0).astype(
        np.uint8
    )
    classes = np.asarray(labels, dtype=np.uint8)
    np.savez(path, x_train=pixels, y_train=classes, x_test=pixels, y_test=classes)


def _run_workflow(images: np.ndarray, labels: np.ndarray, init_seed: int):
    encoder = build_encoder(
        EncodingConfig(n_side=7, power_min=0.01, power_max=0.03)
    )
    geometry = build_geometry(
        GeometryConfig(
            n_side=7,
            pitch_um=1.5,
            sigma_space_um=0.3,
            center_x_um=0.0,
            center_y_um=0.0,
        )
    )
    cfg = Config()
    cfg.grid = GridParameters(nx=32, ny=32, lx=24.0, ly=24.0, grid_type="periodic")
    cfg.compute_engine.use_gpu = False
    cfg.solver.method = "rk4-fdm"
    cfg.solver.precision = "double"
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.012
    cfg.physics.init_seed = init_seed
    cfg.physics.init_mode = "complex_gaussian_zero_mean"
    cfg.physics.init_eps = 1e-3
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.power_definition = "pulse_energy"
    cfg.reservoir.reservoir_type = "single"
    cfg.boundary_condition.absorption = "cap"
    cfg.boundary_condition.mask_width_percent = 0.1
    pulse = PulseConfig(
        sigma_time=0.0015,
        pulse_separation=0.02,
        n_pulses=1,
        cutoff_sigma=2.0,
        power_definition="pulse_energy",
    )
    readout = ReadoutConfig(
        warmup_ps=0.0,
        stride_steps=2,
        mask_radius_um=0.8,
        record_reservoir=False,
        feature_mode="summary",
        batch_size=1,
    )
    powers = encoder.encode(images)
    resources = SharedSimResources(cfg, geometry, pulse, mask_radius_um=0.8)
    bundle = collect_features(resources, powers, labels, readout=readout)
    report = train_and_evaluate(
        bundle,
        seed=17,
        test_fraction=0.5,
        C=1.0,
        max_iter=500,
    )
    return powers, bundle, report


def test_synthetic_digits_end_to_end_is_seeded(tmp_path: Path) -> None:
    """Exercise image loading, encoding, simulation, features, and class readout.

    Block-averaged 7x7 intensities are encoded linearly into 49 pulse energies.
    Per-spot and global condensate-density traces are summarized into float64
    feature vectors. A standardized multinomial logistic regression is trained
    on a stratified half of the synthetic samples, and predictions refer to the
    held-out half. This verifies data flow and seeding, not MNIST accuracy.
    """
    data_path = tmp_path / "synthetic_digits.npz"
    _synthetic_digits(data_path)
    images, labels = load_and_downsample(
        str(data_path), n_samples=20, seed=7, target_side=7
    )
    first_powers, first_bundle, first_report = _run_workflow(images, labels, 41)
    second_powers, second_bundle, second_report = _run_workflow(images, labels, 41)
    _, changed_bundle, _ = _run_workflow(images, labels, 42)

    assert images.shape == (20, 7, 7)
    assert images.dtype == np.float64
    assert labels.shape == (20,)
    assert labels.dtype == np.int64
    assert first_powers.shape == (20, 49)
    assert first_powers.dtype == np.float64
    assert isinstance(first_bundle, FeatureBundle)
    assert first_bundle.features.shape[0] == 20
    assert first_bundle.features.dtype == np.float64
    assert first_bundle.traces_psi.shape[:2] == (20, 6)
    assert first_bundle.traces_psi.dtype == np.float64
    assert isinstance(first_report, ClassificationReport)
    assert first_report.predictions_test.shape == (10,)
    assert first_report.predictions_test.dtype == np.int64
    assert np.array_equal(first_powers, second_powers)
    assert np.array_equal(first_bundle.features, second_bundle.features)
    assert np.array_equal(
        first_report.predictions_test, second_report.predictions_test
    )
    assert not np.array_equal(first_bundle.features, changed_bundle.features)
