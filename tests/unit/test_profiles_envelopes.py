"""Spatial profiles, temporal envelopes and config integration."""
from __future__ import annotations

from math import sqrt

import h5py
import matplotlib.image as mpimg
import numpy as np
import pytest

from polarism import SimulationController, create_envelope, create_profile
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.envelopes import available_envelopes, register_envelope
from polarism.profiles import available_profiles, register_profile
from polarism.profiles.builtin_profiles import _lattice_geometry, _lattice_sites
from polarism.sweep_config import apply_overrides, override_kind
from tests.unit.conftest import grid, small_config


def test_analytic_profiles_and_shapes():
    g = grid(nx=21, ny=21, grid_type="closed-interval")
    np.testing.assert_array_equal(create_profile({"type": "uniform"}, g), np.ones(g.shape))
    gaussian = create_profile({"type": "gaussian", "sigma": 2.0}, g)
    assert gaussian.shape == g.shape
    assert gaussian[10, 10] == pytest.approx(1.0)
    assert gaussian[10, 12] == pytest.approx(np.exp(-0.5))
    supergaussian = create_profile({"type": "supergaussian", "sigma": 2.0, "order": 2.0}, g)
    assert supergaussian[10, 14] == pytest.approx(np.exp(-8.0))
    ring = create_profile({"type": "ring", "radius": 3.0, "width": 1.0}, g)
    assert ring[10, 13] == pytest.approx(1.0)
    assert ring[10, 10] == pytest.approx(np.exp(-4.5))
    shifted = create_profile({"type": "gaussian", "sigma": 2.0, "x0": 1.0, "amplitude": -3.0}, g)
    assert shifted[10, 11] == pytest.approx(-3.0)
    one_d = small_config(grid__ndim=1, grid__nx=21, grid__grid_type="closed-interval")
    from polarism.grid.create_grid import create_grid
    g1 = create_grid(one_d.grid)
    assert create_profile({"type": "gaussian", "sigma": 2.0}, g1).shape == (21,)
    with pytest.raises(ValueError, match="requires a 2D grid"):
        create_profile({"type": "ring", "radius": 2.0, "width": 1.0}, g1)


@pytest.mark.parametrize("suffix", ["npy", "h5", "png"])
def test_file_profile_interpolates_linear_ramp(tmp_path, suffix):
    g = grid(nx=21, ny=21, grid_type="closed-interval")
    xs = np.linspace(-2.0, 2.0, 5)
    ys = np.linspace(-3.0, 3.0, 7)
    data = 0.5 + 0.08 * xs[None, :] + 0.04 * ys[:, None]
    path = tmp_path / f"ramp.{suffix}"
    spec = {"type": "file", "path": str(path), "extent": [-2.0, 2.0, -3.0, 3.0]}
    if suffix == "npy":
        np.save(path, data)
    elif suffix == "h5":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("ramp", data=data)
        spec["dataset"] = "ramp"
    else:
        mpimg.imsave(path, data, cmap="gray", vmin=0.0, vmax=1.0)
    actual = create_profile(spec, g)
    inside = (abs(g.X) <= 2.0) & (abs(g.Y) <= 3.0)
    np.testing.assert_allclose(actual[inside], (0.5 + 0.08 * g.X + 0.04 * g.Y)[inside], atol=0.01 if suffix == "png" else 1e-14)
    assert np.all(actual[~inside] == 0.0)


def test_file_profile_in_one_dimension(tmp_path):
    cfg = small_config(grid__ndim=1, grid__nx=21, grid__grid_type="closed-interval")
    from polarism.grid.create_grid import create_grid
    g = create_grid(cfg.grid)
    path = tmp_path / "line.npy"
    np.save(path, np.linspace(0, 1, 5))
    actual = create_profile({"type": "file", "path": str(path), "extent": [-2, 2]}, g)
    assert actual[10] == pytest.approx(0.5)
    assert actual[0] == 0.0


@pytest.mark.parametrize("kind,count,distance", [
    ("square", 1, 2.0), ("triangular", 1, 2.0),
    ("honeycomb", 2, 2.0 / sqrt(3)), ("kagome", 3, 2.0), ("lieb", 3, 2.0),
])
def test_lattice_unit_cell_and_nearest_neighbour(kind, count, distance):
    basis, offsets = _lattice_geometry(kind, 2.0)
    assert len(offsets) == count
    sites = [i * basis[0] + j * basis[1] + offset
             for i in range(-2, 3) for j in range(-2, 3) for offset in offsets]
    origin = offsets[0]
    distances = [np.linalg.norm(site - origin) for site in sites if np.linalg.norm(site - origin) > 1e-9]
    assert min(distances) == pytest.approx(distance)
    g = grid(nx=21, ny=21, grid_type="closed-interval")
    spec = {"type": "lattice", "lattice": kind, "constant": 2.0, "width": 0.2}
    generated = list(_lattice_sites(g, {**spec, "x0": 0.0, "y0": 0.0}))
    assert sum(i == 0 and j == 0 for i, j, _, _, _ in generated) == count
    result = create_profile(spec, g)
    assert result.shape == g.shape and np.max(result) >= 1.0


def test_lattice_defects_and_disorder():
    g = grid(nx=21, ny=21, grid_type="closed-interval")
    spec = {"type": "lattice", "lattice": "kagome", "constant": 2.0, "width": 0.2}
    original = list(_lattice_sites(g, {**spec, "x0": 0, "y0": 0}))
    defects = [[0, 0, 0], [0, 0, 1]]
    changed = list(_lattice_sites(g, {**spec, "x0": 0, "y0": 0, "defects": defects}))
    assert len(original) - len(changed) == len(defects)
    assert not {(i, j, s) for i, j, s, _, _ in changed}.intersection(map(tuple, defects))
    with_disorder = {**spec, "disorder": 0.1, "seed": 4}
    first = create_profile(with_disorder, g)
    np.testing.assert_array_equal(first, create_profile(with_disorder, g))
    assert not np.array_equal(first, create_profile({**with_disorder, "seed": 5}, g))
    assert not np.array_equal(first, create_profile(spec, g))


def test_envelopes_and_data_files(tmp_path):
    assert create_envelope({"type": "constant"}).time_independent
    step = create_envelope({"type": "step", "t0": 2.0})
    assert not step.time_independent
    assert [step(t) for t in (1.9, 2.0)] == [0.0, 1.0]
    pulses = create_envelope({"type": "gaussian-pulses", "center": 2.0, "sigma": 0.5, "count": 2, "separation": 2.0})
    assert pulses(2.0) == pytest.approx(1.0)
    assert pulses(4.0) == pytest.approx(1.0)
    assert pulses(0.49) == 0.0
    inline = {"type": "data", "times": [0, 2, 4], "values": [1, 3, 5]}
    data = create_envelope(inline)
    assert data(1.0) == 2.0 and data(-1.0) == 0.0 and data(5.0) == 0.0
    hold = create_envelope({**inline, "hold": True})
    assert hold(-1.0) == 1.0 and hold(5.0) == 5.0
    table = np.array([[0, 1], [2, 3], [4, 5]])
    npy = tmp_path / "data.npy"
    csv = tmp_path / "data.csv"
    np.save(npy, table)
    np.savetxt(csv, table, delimiter=",")
    for path in (npy, csv):
        assert create_envelope({"type": "data", "path": str(path)})(1.0) == 2.0


def test_profile_potential_and_laser_controller():
    profile = {"type": "gaussian", "sigma": 2.0, "x0": 1.0}
    cfg = small_config(potential__potential_type="profile", laser__laser_type="profile")
    cfg.potential.profile = {**profile, "amplitude": -2.0}
    cfg.laser.profile = profile
    cfg.laser.envelope = {"type": "step", "t0": 0.03}
    cfg.laser.P0 = 0.4
    cfg.laser.delay = 0.02
    controller = SimulationController(cfg)
    spatial = create_profile(profile, controller.grid)
    np.testing.assert_allclose(controller.potential, -2.0 * spatial)
    assert controller.pump_field(0.04).max() == 0.0
    np.testing.assert_allclose(controller.pump_field(0.05), 0.4 * spatial)
    envelope, coefficient = controller.pump_field.separable_terms()[0]
    np.testing.assert_allclose(envelope, spatial)
    assert coefficient(0.05) == pytest.approx(0.4)
    assert not controller.lasers[0].time_independent


def test_multiple_profile_lasers_and_mapping_overrides(tmp_path):
    path = tmp_path / "lasers.yaml"
    path.write_text("lasers:\n  - laser_type: profile\n    P0: 0.2\n    profile: {type: gaussian, sigma: 2.0}\n    envelope: {type: step, t0: 0.01}\n  - laser_type: profile\n    P0: 0.3\n    profile: {type: uniform}\n    envelope: {type: constant}\n")
    cfg = small_config(laser__mode="multiple", laser__config_file=str(path))
    controller = SimulationController(cfg)
    expected = 0.2 * create_profile({"type": "gaussian", "sigma": 2.0}, controller.grid) + 0.3
    np.testing.assert_allclose(controller.pump_field(0.02), expected)
    changed = apply_overrides(cfg, {"lasers[0].profile.sigma": 3.0, "lasers[0].envelope.t0": 0.02})
    varied = SimulationController(changed)
    assert varied.lasers[0].profile["sigma"] == 3.0
    assert varied.pump_field(0.015).min() == pytest.approx(0.3)


def test_batched_profile_envelope_coefficients():
    cfg = small_config(laser__laser_type="profile", laser__profile={"type": "uniform"})
    cfg.laser.envelope = {"type": "step", "t0": 0.01}
    batch = SimulationController(cfg, members=[{"laser.envelope.t0": 0.01}, {"laser.envelope.t0": 0.03}])
    envelope, coefficient = batch.pump_field.separable_terms()[0]
    assert envelope.shape == batch.grid.shape
    np.testing.assert_allclose(coefficient(0.02), [cfg.laser.P0, 0.0])


@pytest.mark.parametrize("profile,message", [
    ({"type": "unknown"}, "Unknown profile"),
    ({"type": "gaussian", "sigma": 2.0, "wrong": 1}, "Unknown profile key"),
    ({"type": "file"}, "profile.path is required"),
])
def test_profile_validation_errors(profile, message):
    cfg = small_config(potential__potential_type="profile")
    cfg.potential.profile = profile
    with pytest.raises(ConfigValidationError, match=message):
        validate_config(cfg)


def test_envelope_validation_and_power_definition():
    cfg = small_config(laser__laser_type="profile", laser__profile={"type": "uniform"})
    cfg.laser.envelope = {"type": "unknown"}
    with pytest.raises(ConfigValidationError, match="Unknown envelope"):
        validate_config(cfg)
    cfg.laser.envelope = {"type": "constant", "wrong": 1}
    with pytest.raises(ConfigValidationError, match="Unknown envelope key"):
        validate_config(cfg)
    cfg.laser.envelope = {"type": "constant"}
    cfg.laser.power_definition = "pulse_energy"
    with pytest.raises(ConfigValidationError, match="peak_amplitude"):
        validate_config(cfg)


def test_sweep_mapping_paths_and_type_groups():
    assert override_kind("potential.profile.amplitude") == "batch"
    assert override_kind("laser.profile.sigma") == "batch"
    assert override_kind("laser.envelope.t0") == "batch"
    for path in ("potential.profile.type", "laser.profile.type", "laser.envelope.type", "lasers[0].profile.type"):
        assert override_kind(path) == "group"
    cfg = small_config(potential__potential_type="profile", laser__laser_type="profile")
    cfg.potential.profile = {"type": "gaussian", "sigma": 2.0}
    cfg.laser.profile = {"type": "gaussian", "sigma": 2.0}
    cfg.laser.envelope = {"type": "step", "t0": 1.0}
    changed = apply_overrides(cfg, {"potential.profile.sigma": 3.0, "laser.profile.sigma": 4.0, "laser.envelope.t0": 2.0})
    assert changed.potential.profile["sigma"] == 3.0
    assert changed.laser.profile["sigma"] == 4.0
    assert changed.laser.envelope["t0"] == 2.0
    with pytest.raises(ValueError, match="non-batchable"):
        apply_overrides(cfg, {"laser.profile.type": "uniform"})
    grouped = apply_overrides(cfg, {"laser.profile.type": "uniform", "laser.envelope.type": "constant"}, allow_groups=True)
    assert grouped.laser.profile == {"type": "uniform"}
    assert grouped.laser.envelope == {"type": "constant"}


def test_profile_and_envelope_plugins(monkeypatch):
    @register_profile("unit-test-profile", params=("scale",))
    def custom_profile(grid, params, xp):
        return xp.ones(grid.shape) * params["scale"]

    @register_envelope("unit-test-envelope", params=("scale",))
    def custom_envelope(spec):
        return lambda t: spec["scale"] * t

    g = grid()
    assert np.all(create_profile({"type": "unit-test-profile", "scale": 2.0}, g) == 2.0)
    assert create_envelope({"type": "unit-test-envelope", "scale": 2.0})(3.0) == 6.0
    monkeypatch.delitem(available_profiles, "unit-test-profile")
    monkeypatch.delitem(available_envelopes, "unit-test-envelope")
