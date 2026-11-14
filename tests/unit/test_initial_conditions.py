"""Initial-condition registry, built-ins, controller composition and validation."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from polarism.config.simulation_parameters import Config, InitialConditionParameters
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.noise.initial_field import make_initial_psi
from polarism.initial_condition import (
    available_initial_conditions,
    create_initial_condition,
    register_initial_condition,
)
from polarism.simulation_controller import SimulationController
from polarism.simulation_state import SimulationState
from polarism.sweep_config import apply_overrides, override_kind
from tests.unit.conftest import grid, small_config


def make(**changes):
    return InitialConditionParameters(**changes)


def build(g, **changes):
    return create_initial_condition(g, make(**changes), np, np.complex128)


def test_builtins_are_registered() -> None:
    assert {"zero", "plane-wave", "vortex", "file"} <= set(available_initial_conditions)


def test_zero_is_skipped_by_the_factory() -> None:
    assert create_initial_condition(grid(), make(), np, np.complex128) is None
    assert create_initial_condition(grid(), make(type="zero"), np, np.complex128) is None
    zeros = available_initial_conditions.lookup("zero")(grid(), make(), np, np.complex128)
    assert zeros.shape == (16, 16) and not zeros.any()


def test_plane_wave_2d_analytic_form() -> None:
    g = grid()
    psi = build(g, type="plane-wave", amplitude=2.0, kx=0.3, ky=-0.2)
    assert psi.shape == g.shape and psi.dtype == np.complex128
    expected = 2.0 * np.exp(1j * (0.3 * g.X - 0.2 * g.Y))
    np.testing.assert_allclose(psi, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.abs(psi), 2.0, rtol=0, atol=1e-14)


def test_plane_wave_1d_uses_kx_only() -> None:
    cfg = small_config(grid__ndim=1, grid__nx=32, grid__lx=32.0)
    controller = SimulationController(cfg)
    g = controller.grid
    psi = create_initial_condition(g, make(type="plane-wave", kx=0.5, ky=7.0), np, np.complex128)
    assert psi.shape == (32,)
    np.testing.assert_allclose(psi, np.exp(0.5j * g.X), rtol=0, atol=1e-14)


@pytest.mark.parametrize("charge", [1, 2, -1, -3])
def test_vortex_analytic_form(charge) -> None:
    g = grid(32, 32)
    x0, y0, width, amplitude = 0.4, -0.7, 2.5, 1.5
    psi = build(g, type="vortex", amplitude=amplitude, x0=x0, y0=y0, charge=charge, core_width=width)
    dx, dy = g.X - x0, g.Y - y0
    radius = np.hypot(dx, dy)
    expected = amplitude * np.tanh(radius / width) * np.exp(1j * charge * np.arctan2(dy, dx))
    np.testing.assert_allclose(psi, expected, rtol=0, atol=1e-14)
    assert np.abs(psi).max() <= amplitude
    nearest = np.unravel_index(np.argmin(radius), radius.shape)
    assert np.abs(psi[nearest]) < 0.2 * amplitude


@pytest.mark.parametrize("charge", [1, 2, -1, -2])
def test_vortex_phase_winds_by_charge_along_a_loop(charge) -> None:
    g = grid(32, 32)
    psi = build(g, type="vortex", x0=0.3, y0=-0.2, charge=charge, core_width=2.0)
    low, high = 8, 24
    ring = (
        [(low, column) for column in range(low, high)]
        + [(row, high) for row in range(low, high)]
        + [(high, column) for column in range(high, low, -1)]
        + [(row, low) for row in range(high, low, -1)]
    )
    phase = np.angle(np.array([psi[index] for index in ring]))
    steps = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    assert np.isclose(steps.sum() / (2 * np.pi), charge, atol=1e-9)


def test_vortex_loop_not_enclosing_the_core_has_no_winding() -> None:
    g = grid(32, 32)
    psi = build(g, type="vortex", x0=-12.0, y0=-12.0, charge=2, core_width=2.0)
    low, high = 18, 28
    ring = (
        [(low, column) for column in range(low, high)]
        + [(row, high) for row in range(low, high)]
        + [(high, column) for column in range(high, low, -1)]
        + [(row, low) for row in range(high, low, -1)]
    )
    phase = np.angle(np.array([psi[index] for index in ring]))
    steps = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    assert abs(steps.sum()) < 1e-9


def test_file_loads_npy_and_rejects_bad_shape(tmp_path) -> None:
    g = grid()
    rng = np.random.default_rng(1)
    field = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)
    path = tmp_path / "field.npy"
    np.save(path, field)
    psi = build(g, type="file", path=str(path))
    np.testing.assert_array_equal(psi, field)
    np.save(tmp_path / "small.npy", field[:8, :8])
    with pytest.raises(ValueError, match=r"shape \(8, 8\).*grid shape \(16, 16\)"):
        build(g, type="file", path=str(tmp_path / "small.npy"))


def test_file_errors_for_missing_and_unsupported_paths(tmp_path) -> None:
    g = grid()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        build(g, type="file", path=str(tmp_path / "missing.npy"))
    other = tmp_path / "field.txt"
    other.write_text("1")
    with pytest.raises(ValueError, match=r"\.npy, \.h5 or \.hdf5"):
        build(g, type="file", path=str(other))
    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as h5:
        h5.attrs["something"] = 1
    with pytest.raises(ValueError, match="neither a checkpoint v2"):
        build(g, type="file", path=str(empty))


def test_file_loads_results_file_frames(tmp_path) -> None:
    cfg = small_config(result__save_results=True, result__save_hdf5=True, result__save_psi=True,
                       result__save_every=0.02, result__output_directory=str(tmp_path),
                       solver__total_time=0.06)
    controller = SimulationController(cfg)
    controller.run()
    path = controller.run_directory / "results.h5"
    with h5py.File(path, "r") as h5:
        stored = h5["fields/psi"][:]
    assert stored.shape[0] >= 3
    g = controller.grid
    last = build(g, type="file", path=str(path))
    first = build(g, type="file", path=str(path), frame=0)
    middle = build(g, type="file", path=str(path), frame=1)
    np.testing.assert_array_equal(last, stored[-1, 0])
    np.testing.assert_array_equal(first, stored[0, 0])
    np.testing.assert_array_equal(middle, stored[1, 0])
    with pytest.raises(ValueError, match="outside"):
        build(g, type="file", path=str(path), frame=stored.shape[0])


def test_file_loads_member_zero_of_batched_results(tmp_path) -> None:
    g = grid(4, 4)
    stored = np.arange(3 * 2 * 4 * 4, dtype=np.float64).reshape(3, 2, 1, 4, 4) * (1 + 1j)
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as h5:
        dataset = h5.create_dataset("fields/psi", data=stored)
        dataset.attrs["dims"] = ["time", "member", "component", "y", "x"]
    np.testing.assert_array_equal(build(g, type="file", path=str(path)), stored[-1, 0, 0])
    np.testing.assert_array_equal(build(g, type="file", path=str(path), frame=1), stored[1, 0, 0])


def test_file_loads_checkpoint_member_zero_component_zero(tmp_path) -> None:
    source = SimulationController(small_config())
    path = tmp_path / "checkpoint.h5"
    source.save_checkpoint(path)
    loaded = build(source.grid, type="file", path=str(path))
    np.testing.assert_array_equal(loaded, source.state.psi)

    members = [{"noise.seed": 3}, {"noise.seed": 4}]
    batched = SimulationController(small_config(), members=members)
    batched_path = tmp_path / "batched.h5"
    batched.save_checkpoint(batched_path)
    loaded = build(batched.grid, type="file", path=str(batched_path))
    np.testing.assert_array_equal(loaded, batched.state.psi[0])


def test_default_zero_equals_noise_only_field() -> None:
    cfg = small_config()
    controller = SimulationController(cfg)
    expected = make_initial_psi(
        np, controller.grid.shape, controller.grid.spacing, cfg.noise.initial_amplitude,
        mode=cfg.noise.initial, seed=cfg.noise.seed,
    )
    np.testing.assert_array_equal(controller.state.psi, expected)
    explicit = small_config(initial_condition__type="zero")
    np.testing.assert_array_equal(SimulationController(explicit).state.psi, expected)


def test_state_without_condition_matches_zero_condition() -> None:
    g = grid()
    plain = SimulationState(g, 1e-3, "double", seed=5)
    zero = SimulationState(g, 1e-3, "double", seed=5, initial_condition=make(type="zero"))
    np.testing.assert_array_equal(plain.psi, zero.psi)


def test_deterministic_part_is_added_to_noise() -> None:
    cfg = small_config(initial_condition__type="plane-wave", initial_condition__amplitude=0.7,
                       initial_condition__kx=0.2)
    controller = SimulationController(cfg)
    g = controller.grid
    noise = make_initial_psi(np, g.shape, g.spacing, cfg.noise.initial_amplitude,
                             mode=cfg.noise.initial, seed=cfg.noise.seed)
    expected = noise + 0.7 * np.exp(0.2j * g.X)
    np.testing.assert_allclose(controller.state.psi, expected, rtol=0, atol=1e-14)


def test_disabled_noise_leaves_only_the_deterministic_part() -> None:
    cfg = small_config(noise__enabled=False, initial_condition__type="vortex",
                       initial_condition__charge=-1)
    controller = SimulationController(cfg)
    g = controller.grid
    np.testing.assert_allclose(
        controller.state.psi, build(g, type="vortex", charge=-1), rtol=0, atol=1e-14,
    )


def test_single_precision_state_dtype() -> None:
    cfg = small_config(solver__precision="single", initial_condition__type="plane-wave",
                       noise__enabled=False)
    controller = SimulationController(cfg)
    assert controller.state.psi.dtype == np.complex64


def test_batched_members_get_their_own_deterministic_part() -> None:
    cfg = small_config(initial_condition__type="plane-wave", initial_condition__kx=0.3)
    members = [{"initial_condition.amplitude": 1.0}, {"initial_condition.amplitude": 2.0},
               {"initial_condition.kx": 0.6}]
    batched = SimulationController(cfg, members=members)
    assert batched.state.psi.shape == (3, 16, 16)
    for index, override in enumerate(members):
        single_cfg = small_config(initial_condition__type="plane-wave", initial_condition__kx=0.3)
        for path, value in override.items():
            setattr(single_cfg.initial_condition, path.split(".")[1], value)
        np.testing.assert_array_equal(
            batched.state.psi[index], SimulationController(single_cfg).state.psi,
        )


def test_plugin_condition_is_used_and_shape_is_checked() -> None:
    @register_initial_condition("dummy-gaussian")
    def gaussian(grid, cfg, xp, cdtype):
        return cfg.amplitude * xp.exp(-(grid.X ** 2 + grid.Y ** 2) / cfg.core_width ** 2)

    @register_initial_condition("dummy-wrong-shape")
    def wrong(grid, cfg, xp, cdtype):
        return xp.zeros((3,), dtype=cdtype)

    try:
        cfg = small_config(initial_condition__type="dummy-gaussian", noise__enabled=False,
                           initial_condition__core_width=4.0)
        controller = SimulationController(cfg)
        g = controller.grid
        np.testing.assert_allclose(controller.state.psi, np.exp(-(g.X ** 2 + g.Y ** 2) / 16.0),
                                   rtol=0, atol=1e-14)
        with pytest.raises(ValueError, match="returned shape"):
            SimulationController(small_config(initial_condition__type="dummy-wrong-shape"))
    finally:
        available_initial_conditions.pop("dummy-gaussian", None)
        available_initial_conditions.pop("dummy-wrong-shape", None)


@pytest.mark.parametrize("path,value,message", [
    ("initial_condition__type", "missing", "Available"),
    ("initial_condition__type", 3, "must be a string"),
    ("initial_condition__amplitude", -1.0, "amplitude must be >= 0"),
    ("initial_condition__amplitude", float("nan"), "finite number"),
    ("initial_condition__kx", float("inf"), "finite number"),
    ("initial_condition__x0", True, "finite number"),
    ("initial_condition__core_width", 0.0, "core_width must be > 0"),
    ("initial_condition__charge", 1.5, "charge must be an integer"),
    ("initial_condition__frame", "last", "frame must be an integer"),
    ("initial_condition__path", "", "nonempty path"),
    ("initial_condition__extra", [], "must be a mapping"),
])
def test_invalid_initial_condition_values(path, value, message) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(small_config(**{path: value}))


def test_file_type_requires_a_path() -> None:
    with pytest.raises(ConfigValidationError, match="path is required"):
        validate_config(small_config(initial_condition__type="file"))


def test_vortex_is_rejected_in_1d() -> None:
    cfg = small_config(grid__ndim=1, initial_condition__type="vortex")
    with pytest.raises(ConfigValidationError, match="requires a 2D grid"):
        validate_config(cfg)
    cfg.initial_condition.type = "plane-wave"
    validate_config(cfg)


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("initial_condition:\n  type: vortex\n  charge: -2\n  core_width: 2.0\n")
    cfg = Config.from_yaml(path)
    assert cfg.initial_condition.type == "vortex"
    assert cfg.initial_condition.charge == -2
    assert cfg.initial_condition.core_width == 2.0
    path.write_text("initial_condition:\n  shape: vortex\n")
    with pytest.raises(ValueError, match="Unknown config key 'initial_condition.shape'"):
        Config.from_yaml(path)


def test_sweep_path_kinds() -> None:
    for field in ("amplitude", "x0", "y0", "kx", "ky", "charge", "core_width", "frame", "extra"):
        assert override_kind(f"initial_condition.{field}") == "batch"
    assert override_kind("initial_condition.extra.width") == "batch"
    assert override_kind("initial_condition.type") == "group"
    assert override_kind("initial_condition.path") == "group"
    with pytest.raises(ValueError, match="Unknown sweep path"):
        override_kind("initial_condition.missing")


def test_apply_overrides_for_initial_condition() -> None:
    cfg = small_config(initial_condition__type="plane-wave")
    changed = apply_overrides(cfg, {"initial_condition.amplitude": 3.0})
    assert changed.initial_condition.amplitude == 3.0 and cfg.initial_condition.amplitude == 1.0
    with pytest.raises(ValueError, match="non-batchable"):
        apply_overrides(cfg, {"initial_condition.type": "vortex"})
    assert apply_overrides(cfg, {"initial_condition.type": "vortex"},
                           allow_groups=True).initial_condition.type == "vortex"
    with pytest.raises(ConfigValidationError, match="amplitude must be >= 0"):
        apply_overrides(cfg, {"initial_condition.amplitude": -1.0})
