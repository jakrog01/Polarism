"""Canonical internal layout and public scalar views."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from polarism import Config, SimulationController
from polarism.compute_engine import compute_engine
from polarism.config.batch_physics import BatchPhysics
from polarism.config.simulation_parameters import ComputeEngineParameters, PhysicsConstants
from polarism.grid.create_grid import create_grid
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState


def _config():
    cfg = Config()
    cfg.grid.nx = 8
    cfg.grid.ny = 6
    cfg.result.save_results = False
    cfg.compute_engine = ComputeEngineParameters(use_gpu=False)
    return cfg


@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_state_canonical_and_public_views(batch_size):
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    grid = create_grid(_config().grid)
    state = SimulationState(grid, seed=7, batch_size=batch_size)
    size = 1 if batch_size is None else batch_size
    public = (6, 8) if batch_size is None else (size, 6, 8)
    assert state.batch_size == size
    assert state.components == 1
    assert state.batched is (batch_size is not None)
    assert state.field.shape == (size, 1, 6, 8)
    assert state.psi.shape == public
    assert np.shares_memory(state.psi, state.field)
    state.field_k = np.fft.fft2(state.field, axes=(-2, -1))
    assert state.psi_k.shape == public
    assert np.shares_memory(state.psi_k, state.field_k)
    replacement = np.ones(public, dtype=np.complex128)
    state.psi = replacement
    assert state.field_k is None
    assert np.shares_memory(state.psi, replacement)
    state.psi_k = np.fft.fft2(state.psi, axes=(-2, -1))
    state.field = None
    state.psi_k = np.fft.fft2(replacement, axes=(-2, -1))
    assert np.array_equal(state.psi, replacement)
    assert state.field_k is not None
    assert np.shares_memory(state.psi, state.field)
    with pytest.raises(ValueError, match="field shape"):
        state.field = np.zeros(public, dtype=np.complex128)
    with pytest.raises(ValueError, match="field_k shape"):
        state.field_k = np.zeros(public, dtype=np.complex128)
    with pytest.raises(ValueError, match="psi shape"):
        state.psi = np.zeros((4, 4), dtype=np.complex128)


@pytest.mark.parametrize("model,fields", [("single", 1), ("double", 2),
                                         ("quadratic-double", 2)])
def test_reservoir_tuples_are_canonical(model, fields):
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    cfg = _config()
    cfg.reservoir.reservoir_type = model
    grid = create_grid(cfg.grid)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid, batch_size=2)
    state = reservoir.get_state()
    assert len(state) == fields
    assert all(value.shape == (2, 1, 6, 8) for value in state)
    with pytest.raises(ValueError, match="reservoir state shape"):
        reservoir.set_state(tuple(np.zeros((6, 8)) for _ in state))
    derivatives = reservoir.get_derivatives(
        np.ones((2, 1, 6, 8), dtype=np.complex128), np.ones((6, 8)), state
    )
    assert all(value.shape == (2, 1, 6, 8) for value in derivatives)
    assert reservoir.get_reservoir_density().shape == (2, 6, 8)
    reservoir.reset()
    assert all(value.shape == (1, 1, 6, 8) for value in reservoir.get_state())
    assert reservoir.get_reservoir_density().shape == (6, 8)


def test_batch_physics_broadcasts_on_member_axis():
    values = {"g_C": [0.2, 0.4]}
    physics = BatchPhysics(PhysicsConstants(), values, np)
    field = np.ones((2, 1, 6, 8))
    physics.check(field)
    assert physics.get("g_C").shape == (2, 1, 1, 1)
    np.testing.assert_array_equal(field * physics.get("g_C")[:, :, 0:1, 0:1],
                                  np.broadcast_to(np.asarray(values["g_C"])[:, None, None, None],
                                                  field.shape))
    with pytest.raises(ValueError, match="canonical"):
        physics.check(np.ones((2, 6, 8)))
    with pytest.raises(ValueError, match="state batch size"):
        physics.check(np.ones((1, 1, 6, 8)))


def test_initial_state_can_add_public_batch_axis():
    cfg = _config()
    cfg.reservoir.reservoir_type = "single"
    controller = SimulationController(cfg)
    psi = np.ones((2, 6, 8), dtype=np.complex128)
    density = np.ones((2, 6, 8), dtype=np.float64)
    controller.set_initial_state(psi, (density,))
    assert controller.state.batched
    assert controller.state.field.shape == (2, 1, 6, 8)
    assert controller.reservoir.get_state()[0].shape == (2, 1, 6, 8)
    assert controller.reservoir.get_reservoir_density().shape == (2, 6, 8)


@pytest.mark.parametrize("batched", [False, True])
def test_public_initial_state_and_v1_checkpoint(tmp_path, batched):
    cfg = _config()
    cfg.reservoir.reservoir_type = "single"
    members = [{}, {}] if batched else None
    first = SimulationController(cfg, members=members)
    public_shape = (2, 6, 8) if batched else (6, 8)
    psi = np.arange(np.prod(public_shape), dtype=np.float64).reshape(public_shape).astype(np.complex128)
    density = np.ones(public_shape, dtype=np.float64)
    first.set_initial_state(psi, (density,))
    assert first.state.field.shape == ((2 if batched else 1), 1, 6, 8)
    assert first.reservoir.get_state()[0].shape == first.state.field.shape
    np.testing.assert_array_equal(first.state.psi, psi)
    np.testing.assert_array_equal(first.reservoir.get_reservoir_density(), density)
    first.state.psi_k = np.fft.fft2(first.state.psi, axes=(-2, -1))
    path = tmp_path / "checkpoint.h5"
    first.save_checkpoint(path)
    with h5py.File(path, "r") as saved:
        assert saved.attrs["checkpoint_version"] == 1
        assert saved["psi"].shape == public_shape
        assert saved["psi_k"].shape == public_shape
        assert saved["reservoir"]["0"].shape == public_shape
    second = SimulationController(cfg, members=members)
    second.load_checkpoint(path)
    np.testing.assert_array_equal(second.state.psi, psi)
    np.testing.assert_array_equal(second.state.psi_k, first.state.psi_k)
    np.testing.assert_array_equal(second.reservoir.get_reservoir_density(), density)
