"""Built-in frame observables: analytic values, batching and validation."""
from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from polarism.config.validation import ConfigValidationError, validate_config
from polarism.results.observable_registry import available_observables
from polarism.results.storage.base_storage import BaseStorage
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config

SIZE = 64
LENGTH = 32.0
CORE = SIZE // 2 - 1
BUILTINS = ("angular-momentum", "center-of-mass", "current", "momentum-density", "vortices")
MEMBERS = [{"noise.seed": 1}, {"noise.seed": 2}]


def config(**overrides):
    base = dict(noise__enabled=False, grid__nx=SIZE, grid__ny=SIZE, grid__lx=LENGTH, grid__ly=LENGTH)
    base.update(overrides)
    return small_config(**base)


def config_1d():
    return small_config(noise__enabled=False, grid__ndim=1, grid__nx=SIZE, grid__lx=LENGTH)


def build(name, cfg=None, members=None):
    controller = SimulationController(cfg if cfg is not None else config(), members=members)
    produced = available_observables.lookup(name)(controller)
    nodes = produced if isinstance(produced, list) else [produced]
    return controller, {node.name: node for node in nodes}


def evaluate(controller, node, psi):
    controller.state.psi = np.asarray(psi, dtype=np.result_type(psi, np.complex64))
    return node.compute_fn(state=controller.state, grid=controller.grid)


def plane_wave(grid, kx, ky=0.0, amplitude=1.0):
    return amplitude * np.exp(1j * (kx * grid.X + ky * grid.Y))


def vortex(grid, charge, width=1.5, envelope=None, x0=0.0, y0=0.0):
    dx, dy = grid.X - x0, grid.Y - y0
    psi = np.tanh(np.hypot(dx, dy) / width) * np.exp(1j * charge * np.arctan2(dy, dx))
    if envelope is not None:
        psi = psi * np.exp(-(grid.X ** 2 + grid.Y ** 2) / (2 * envelope ** 2))
    return psi


def norm(psi, grid):
    return (np.abs(psi) ** 2).sum(axis=(-2, -1)) * grid.spacing[0] * grid.spacing[1]


def test_builtins_are_registered_in_sorted_order_without_a_norm_duplicate() -> None:
    names = [name for name in available_observables if name in BUILTINS]
    assert names == sorted(BUILTINS)
    assert "norm" not in available_observables


def test_node_metadata_and_kinds() -> None:
    expected = {
        "center-of-mass": {"center_x": False, "center_y": False},
        "momentum-density": {"momentum_density": True},
        "current": {"current_x": True, "current_y": True},
        "angular-momentum": {"angular_momentum": False},
        "vortices": {"vortex_charge": True, "vortex_count": False},
    }
    for observable, kinds in expected.items():
        _, nodes = build(observable)
        assert {name: node.is_field for name, node in nodes.items()} == kinds
        for node in nodes.values():
            assert node.save and node.units is not None and node.label
            assert node.expose is (not node.is_field)
            assert node.cmap is None
    assert build("angular-momentum")[1]["angular_momentum"].units == "ħ"
    assert build("center-of-mass")[1]["center_x"].units == "µm"
    assert build("current")[1]["current_x"].units == "µm^-1 ps^-1"


def test_plane_wave_momentum_density_peaks_at_k_and_integrates_to_the_norm() -> None:
    controller, nodes = build("momentum-density")
    g = controller.grid
    dk = 2 * math.pi / LENGTH
    psi = plane_wave(g, 3 * dk, -2 * dk, amplitude=1.5)
    density = evaluate(controller, nodes["momentum_density"], psi)
    assert density.shape == g.shape
    assert np.unravel_index(np.argmax(density), density.shape) == (SIZE // 2 - 2, SIZE // 2 + 3)
    assert np.count_nonzero(density > 1e-12 * density.max()) == 1
    np.testing.assert_allclose(density.sum() * dk * dk, norm(psi, g), rtol=1e-12)


def test_gaussian_momentum_density_matches_the_analytic_transform() -> None:
    controller, nodes = build("momentum-density")
    g = controller.grid
    width = 2.0
    psi = np.exp(-(g.X ** 2 + g.Y ** 2) / (2 * width ** 2))
    density = evaluate(controller, nodes["momentum_density"], psi)
    k = 2 * math.pi * np.fft.fftshift(np.fft.fftfreq(SIZE, d=LENGTH / SIZE))
    KX, KY = np.meshgrid(k, k, indexing="xy")
    expected = width ** 4 * np.exp(-width ** 2 * (KX ** 2 + KY ** 2))
    np.testing.assert_allclose(density, expected, rtol=0, atol=1e-9 * expected.max())


def test_plane_wave_current_is_hbar_k_over_m_times_density() -> None:
    controller, nodes = build("current")
    g = controller.grid
    physics = controller.cfg.physics
    k = 2 * math.pi / LENGTH
    amplitude = 1.3
    psi = plane_wave(g, k, amplitude=amplitude)
    jx = evaluate(controller, nodes["current_x"], psi)
    jy = evaluate(controller, nodes["current_y"], psi)
    h = g.spacing[1]
    expected = physics.hbar / physics.m_eff * amplitude ** 2 * math.sin(k * h) / h
    assert jx.shape == g.shape
    np.testing.assert_allclose(jx, expected, rtol=1e-12)
    np.testing.assert_allclose(jx, physics.hbar * k / physics.m_eff * amplitude ** 2, rtol=5e-3)
    np.testing.assert_allclose(jy, 0.0, atol=1e-12)


def test_current_on_closed_interval_grid_is_uniform_up_to_the_edges() -> None:
    controller, nodes = build("current", config(grid__grid_type="closed-interval"))
    g = controller.grid
    physics = controller.cfg.physics
    ky = 0.4
    psi = plane_wave(g, 0.0, ky)
    jy = evaluate(controller, nodes["current_y"], psi)
    jx = evaluate(controller, nodes["current_x"], psi)
    h = g.spacing[0]
    np.testing.assert_allclose(jy, physics.hbar / physics.m_eff * math.sin(ky * h) / h, rtol=1e-12)
    np.testing.assert_allclose(jx, 0.0, atol=1e-12)


def test_plane_wave_has_zero_angular_momentum() -> None:
    controller, nodes = build("angular-momentum")
    psi = plane_wave(controller.grid, 2 * math.pi / LENGTH, 4 * math.pi / LENGTH)
    value = evaluate(controller, nodes["angular_momentum"], psi)
    assert abs(float(value)) < 1e-10


@pytest.mark.parametrize("charge", [1, 2, -1])
def test_vortex_angular_momentum_per_particle_equals_the_charge(charge) -> None:
    controller, nodes = build("angular-momentum")
    psi = vortex(controller.grid, charge, envelope=4.0)
    value = evaluate(controller, nodes["angular_momentum"], psi)
    assert float(value) == pytest.approx(charge, abs=0.05)


def test_angular_momentum_sign_follows_the_phase_winding_direction() -> None:
    controller, nodes = build("angular-momentum")
    psi = vortex(controller.grid, 1, envelope=4.0)
    assert float(evaluate(controller, nodes["angular_momentum"], psi)) > 0.9
    assert float(evaluate(controller, nodes["angular_momentum"], psi.conj())) < -0.9


@pytest.mark.parametrize("charge", [1, -1])
def test_vortex_is_detected_with_its_charge_at_the_core(charge) -> None:
    controller, nodes = build("vortices")
    psi = vortex(controller.grid, charge, envelope=4.0)
    winding = evaluate(controller, nodes["vortex_charge"], psi)
    count = evaluate(controller, nodes["vortex_count"], psi)
    assert winding.dtype == np.int8 and winding.shape == controller.grid.shape
    assert winding[CORE, CORE] == charge
    assert int(count) == 1


def test_off_centre_vortex_is_found_in_the_plaquette_containing_the_core() -> None:
    controller, nodes = build("vortices")
    g = controller.grid
    psi = vortex(g, 1, envelope=4.0, x0=3.1, y0=-2.2)
    winding = evaluate(controller, nodes["vortex_charge"], psi)
    column = int(np.floor(3.1 / g.spacing[1] + (SIZE - 1) / 2))
    row = int(np.floor(-2.2 / g.spacing[0] + (SIZE - 1) / 2))
    assert winding[row, column] == 1
    assert int(evaluate(controller, nodes["vortex_count"], psi)) == 1


def test_closed_interval_charge_field_sums_to_the_total_charge() -> None:
    controller, nodes = build("vortices", config(grid__grid_type="closed-interval"))
    psi = vortex(controller.grid, -1)
    winding = evaluate(controller, nodes["vortex_charge"], psi)
    assert int(winding.sum()) == -1
    assert winding[CORE, CORE] == -1
    assert not winding[-1, :].any() and not winding[:, -1].any()
    assert int(evaluate(controller, nodes["vortex_count"], psi)) == 1


def test_vortex_antivortex_pair_counts_two() -> None:
    controller, nodes = build("vortices")
    g = controller.grid
    first = vortex(g, 1, x0=-5.0)
    second = vortex(g, -1, x0=5.0)
    psi = first * second * np.exp(-(g.X ** 2 + g.Y ** 2) / 50.0)
    winding = evaluate(controller, nodes["vortex_charge"], psi)
    row = CORE
    assert winding[row, 21] == 1 and winding[row, 41] == -1
    assert int(evaluate(controller, nodes["vortex_count"], psi)) == 2


def test_phase_noise_in_empty_regions_is_not_counted() -> None:
    controller, nodes = build("vortices", config(grid__grid_type="closed-interval"))
    g = controller.grid
    rng = np.random.default_rng(0)
    noise = 1e-3 * (rng.normal(size=g.shape) + 1j * rng.normal(size=g.shape))
    psi = vortex(g, 1, envelope=4.0) + noise
    assert np.count_nonzero(evaluate(controller, nodes["vortex_charge"], psi)) > 2
    assert int(evaluate(controller, nodes["vortex_count"], psi)) == 1


def test_smooth_field_without_vortices_has_none() -> None:
    controller, nodes = build("vortices")
    psi = plane_wave(controller.grid, 2 * math.pi / LENGTH, 2 * math.pi / LENGTH)
    assert not evaluate(controller, nodes["vortex_charge"], psi).any()
    assert int(evaluate(controller, nodes["vortex_count"], psi)) == 0


def test_center_of_mass_of_a_displaced_gaussian() -> None:
    controller, nodes = build("center-of-mass")
    g = controller.grid
    psi = np.exp(-((g.X - 1.5) ** 2 + (g.Y + 2.25) ** 2) / (2 * 2.0 ** 2))
    assert float(evaluate(controller, nodes["center_x"], psi)) == pytest.approx(1.5, abs=1e-6)
    assert float(evaluate(controller, nodes["center_y"], psi)) == pytest.approx(-2.25, abs=1e-6)


def test_center_of_mass_of_an_empty_field_is_finite() -> None:
    controller, nodes = build("center-of-mass")
    value = evaluate(controller, nodes["center_x"], np.zeros(controller.grid.shape, dtype=complex))
    assert float(value) == 0.0


@pytest.mark.parametrize("name", BUILTINS)
def test_batched_scalars_and_fields_keep_the_member_axis(name) -> None:
    controller, nodes = build(name, members=MEMBERS)
    single, single_nodes = build(name)
    g = controller.grid
    psis = np.stack([
        vortex(g, 1, envelope=4.0) * np.exp(-((g.X - 1.0) ** 2) / 8.0),
        vortex(g, -1, envelope=4.0) * np.exp(-((g.X + 1.0) ** 2) / 8.0),
    ])
    for node_name, node in nodes.items():
        value = evaluate(controller, node, psis)
        if node.is_field:
            assert value.shape == (2, *g.shape)
        else:
            assert value.shape == (2,)
        for member in range(2):
            reference = evaluate(single, single_nodes[node_name], psis[member])
            np.testing.assert_allclose(value[member], reference, rtol=1e-12, atol=1e-12)


def test_batched_vortex_and_angular_momentum_values_per_member() -> None:
    controller, nodes = build("vortices", members=MEMBERS)
    g = controller.grid
    psis = np.stack([vortex(g, 1, envelope=4.0), vortex(g, -1, envelope=4.0)])
    charge = evaluate(controller, nodes["vortex_charge"], psis)
    assert charge.dtype == np.int8
    assert charge[0, CORE, CORE] == 1 and charge[1, CORE, CORE] == -1
    np.testing.assert_array_equal(evaluate(controller, nodes["vortex_count"], psis), [1, 1])
    controller, nodes = build("angular-momentum", members=MEMBERS)
    np.testing.assert_allclose(evaluate(controller, nodes["angular_momentum"], psis), [1, -1], atol=0.05)


def test_batched_current_uses_per_member_physics() -> None:
    controller, nodes = build("current", members=[{"physics.hbar": 0.5}, {"physics.hbar": 1.0}])
    psi = plane_wave(controller.grid, 2 * math.pi / LENGTH)
    current = evaluate(controller, nodes["current_x"], np.stack([psi, psi]))
    np.testing.assert_allclose(current[1], 2 * current[0], rtol=1e-12)


@pytest.mark.parametrize("name", BUILTINS)
def test_single_precision_keeps_the_working_dtype(name) -> None:
    controller, nodes = build(name, config(solver__precision="single"))
    psi = vortex(controller.grid, 1, envelope=4.0).astype(np.complex64)
    for node_name, node in nodes.items():
        value = evaluate(controller, node, psi)
        if node_name == "vortex_charge":
            assert value.dtype == np.int8
        elif node_name == "vortex_count":
            assert np.issubdtype(value.dtype, np.integer)
        else:
            assert value.dtype == np.float32


def test_one_dimensional_observables() -> None:
    cfg = config_1d()
    k = 2 * math.pi / LENGTH

    controller, nodes = build("center-of-mass", cfg)
    assert list(nodes) == ["center_x"]
    g = controller.grid
    psi = np.exp(-((g.X - 2.0) ** 2) / 8.0)
    assert float(evaluate(controller, nodes["center_x"], psi)) == pytest.approx(2.0, abs=1e-6)

    controller, nodes = build("current", cfg)
    assert list(nodes) == ["current_x"] and nodes["current_x"].units == "ps^-1"
    physics = controller.cfg.physics
    jx = evaluate(controller, nodes["current_x"], np.exp(1j * k * g.X))
    assert jx.shape == (SIZE,)
    np.testing.assert_allclose(jx, physics.hbar * k / physics.m_eff, rtol=5e-3)

    controller, nodes = build("momentum-density", cfg)
    psi = np.exp(3j * k * g.X)
    density = evaluate(controller, nodes["momentum_density"], psi)
    assert density.shape == (SIZE,) and int(np.argmax(density)) == SIZE // 2 + 3
    np.testing.assert_allclose(density.sum() * k, (np.abs(psi) ** 2).sum() * g.spacing[0], rtol=1e-12)


@pytest.mark.parametrize("name", ["angular-momentum", "vortices"])
def test_two_dimensional_observables_are_rejected_for_1d_grids(name) -> None:
    cfg = config_1d()
    cfg.result.observables = (name,)
    with pytest.raises(ConfigValidationError, match=f"{name}.*2D"):
        validate_config(cfg)


@pytest.mark.parametrize("name", ["center-of-mass", "current", "momentum-density"])
def test_dimension_agnostic_observables_validate_in_1d(name) -> None:
    cfg = config_1d()
    cfg.result.observables = (name,)
    validate_config(cfg)


def test_momentum_density_dataset_dims_use_k_axes(tmp_path) -> None:
    class Storage(BaseStorage):
        def dump_batch(self): ...
        def write_static(self, name, field): ...
        def finalize(self): ...

    storage = Storage(tmp_path, 4)
    assert storage.dataset_metadata("momentum_density", field=True)["dims"] == ["time", "ky", "kx"]
    assert storage.dataset_metadata("density", field=True)["dims"] == ["time", "y", "x"]
    storage.ndim = 1
    assert storage.dataset_metadata("momentum_density", field=True)["dims"] == ["time", "kx"]


def test_run_saves_only_the_requested_observables(tmp_path) -> None:
    cfg = config(solver__total_time=0.04, result__save_results=True, result__save_hdf5=True,
                 result__save_every=0.02, result__output_directory=str(tmp_path),
                 initial_condition__type="vortex", initial_condition__core_width=1.5)
    cfg.result.observables = ("vortices", "center-of-mass")
    controller = SimulationController(cfg)
    controller.run()
    with h5py.File(controller.run_directory / "results.h5", "r") as h5:
        assert "vortex_charge" in h5["fields"]
        assert {"norm", "vortex_count", "center_x", "center_y"} <= set(h5["scalars"])
        assert "current_x" not in h5["fields"] and "angular_momentum" not in h5["scalars"]
        assert h5["fields/vortex_charge"].dtype == np.int8
        assert h5["scalars/center_x"].shape == h5["time"].shape


def test_default_run_adds_no_observable_nodes() -> None:
    names = {node.name for node in SimulationController(config())._build_result_nodes()}
    assert "norm" in names
    assert names.isdisjoint({"center_x", "center_y", "momentum_density", "current_x", "current_y",
                             "angular_momentum", "vortex_charge", "vortex_count"})
