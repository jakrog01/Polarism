"""Run events, steady-state checks, and per-member stop times."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from polarism.config.validation import ConfigValidationError, validate_config
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def quiet_config(**overrides):
    """Build a short run with a decaying condensate and no pump."""
    values = {"noise__enabled": False, "reservoir__reservoir_type": "none",
              "laser__P0": 0.0, "initial_condition__type": "plane-wave",
              "initial_condition__amplitude": 1e-17, "solver__total_time": 0.1}
    values.update(overrides)
    cfg = small_config(**values)
    cfg.checkpoint.on_signal = False
    return cfg


def test_steady_state_near_zero_requires_two_checks():
    cfg = quiet_config(stop__steady_state=True, stop__check_every=0.02)
    controller = SimulationController(cfg)
    controller.run()
    assert controller.state.step == 4
    np.testing.assert_allclose(controller.stop_times, [0.04])


def test_user_predicate_stops_at_exact_callback_time():
    controller = SimulationController(quiet_config())
    observed = []

    def stop_at_six(view):
        observed.append((view.step, view.t, view.batched, view.batch_size))
        assert not view.psi.flags.writeable
        assert not view.density.flags.writeable
        assert all(not field.flags.writeable for field in view.reservoir)
        return view.step == 6

    controller.add_event(stop_at_six, 0.03)
    controller.run()
    assert [entry[0] for entry in observed] == [3, 6]
    assert all(entry[2:] == (False, 1) for entry in observed)
    np.testing.assert_allclose(controller.stop_times, [0.06])
    assert controller.state.step == 6


@pytest.mark.parametrize("fmt", ["hdf5", "json", "npy"])
def test_batched_member_stop_times_and_storage(tmp_path, fmt):
    cfg = quiet_config(solver__total_time=0.06)
    cfg.result.save_results = True
    cfg.result.formats = (fmt,)
    members = [{"initial_condition.amplitude": 1e-17},
               {"initial_condition.amplitude": 2e-17}]
    controller = SimulationController(cfg, members=members, output_dir=tmp_path)

    def stop_individually(view):
        assert view.psi.shape[0] == 2
        assert view.batched and view.batch_size == 2
        return np.array([view.step == 2, view.step == 4])

    controller.add_event(stop_individually, 0.01)
    controller.run()
    assert controller.state.step == 4
    np.testing.assert_allclose(controller.stop_times, [0.02, 0.04])
    if fmt == "hdf5":
        with h5py.File(tmp_path / "results.h5") as handle:
            np.testing.assert_allclose(handle.attrs["stop_time"], [0.02, 0.04])
    else:
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        np.testing.assert_allclose(metadata["stop_time"], [0.02, 0.04])


def test_event_clips_solver_chunks():
    controller = SimulationController(quiet_config())
    chunks = []
    advance = controller.solver.advance

    def tracked(model, state, count):
        chunks.append(count)
        advance(model, state, count)

    controller.solver.advance = tracked
    callback_steps = []
    controller.add_event(lambda view: callback_steps.append(view.step), 0.03)
    controller.run()
    assert chunks == [3, 3, 3, 1]
    assert callback_steps == [3, 6, 9]


def test_unstopped_member_has_nan_metadata(tmp_path):
    cfg = quiet_config(solver__total_time=0.02)
    cfg.result.save_results = True
    cfg.result.formats = ("hdf5",)
    controller = SimulationController(cfg, output_dir=tmp_path)
    controller.run()
    with h5py.File(tmp_path / "results.h5") as handle:
        assert handle.attrs["stop_time"].shape == (1,)
        assert np.isnan(handle.attrs["stop_time"][0])


def test_checkpoint_restores_steady_reference_and_stop_times(tmp_path):
    cfg = quiet_config(solver__total_time=0.02, stop__steady_state=True,
                       stop__check_every=0.02)
    cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    first = SimulationController(cfg)
    first.run()
    assert first._steady_last_step == 2
    assert np.isnan(first.stop_times[0])
    with h5py.File(cfg.checkpoint.path) as handle:
        assert handle.attrs["steady_last_step"] == 2
        assert handle["steady_density"].shape == first.state.psi.shape
        assert np.isnan(handle.attrs["stop_time"][0])
    resumed_cfg = quiet_config(solver__total_time=0.06, stop__steady_state=True,
                               stop__check_every=0.02)
    resumed_cfg.checkpoint.resume_from = cfg.checkpoint.path
    resumed = SimulationController(resumed_cfg)
    resumed.run()
    assert resumed.state.step == 4
    np.testing.assert_allclose(resumed.stop_times, [0.04])


def test_checkpoint_keeps_first_member_stop_when_second_stops_later(tmp_path):
    members = [{"initial_condition.amplitude": 1e-17},
               {"initial_condition.amplitude": 2e-17}]
    cfg = quiet_config(solver__total_time=0.02)
    cfg.checkpoint.path = str(tmp_path / "checkpoint.h5")
    first = SimulationController(cfg, members=members)
    first.add_event(lambda view: np.array([view.step == 2, False]), 0.01)
    first.run()
    np.testing.assert_allclose(first.stop_times[0], 0.02)
    assert np.isnan(first.stop_times[1])
    resumed_cfg = quiet_config(solver__total_time=0.06)
    resumed_cfg.checkpoint.resume_from = cfg.checkpoint.path
    resumed = SimulationController(resumed_cfg, members=members)
    resumed.add_event(lambda view: np.array([False, view.step == 4]), 0.01)
    resumed.run()
    assert resumed.state.step == 4
    np.testing.assert_allclose(resumed.stop_times, [0.02, 0.04])


def test_default_run_keeps_original_chunk_boundary():
    controller = SimulationController(quiet_config())
    chunks = []
    advance = controller.solver.advance

    def tracked(model, state, count):
        chunks.append(count)
        advance(model, state, count)

    controller.solver.advance = tracked
    controller.run()
    assert chunks == [10]
    assert np.isnan(controller.stop_times[0])


@pytest.mark.parametrize("field,value", [
    ("steady_state", 1), ("rtol", 0.0), ("rtol", float("nan")),
    ("check_every", 0.015), ("min_time", -0.01),
])
def test_stop_validation(field, value):
    cfg = quiet_config(stop__steady_state=True)
    setattr(cfg.stop, field, value)
    with pytest.raises(ConfigValidationError, match=f"stop.{field}"):
        validate_config(cfg)


def test_event_interval_validation():
    controller = SimulationController(quiet_config())
    with pytest.raises(ConfigValidationError, match="event every"):
        controller.add_event(lambda view: None, 0.015)
