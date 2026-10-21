"""Sweep storage, reader, and checkpoint integration tests."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from polarism import SimulationController, Sweep, load_results
from tests.unit.conftest import small_config


def sweep_config(tmp_path, backend="hdf5"):
    """Build a small output enabled sweep configuration."""
    cfg = small_config(result__save_results=True,
                       result__output_directory=str(tmp_path),
                       solver__total_time=0.04)
    setattr(cfg.result, f"save_{backend}", True)
    cfg.result.save_psi = True
    cfg.sweep.batch_size = 2
    return cfg


@pytest.mark.parametrize("backend", ["hdf5", "npy"])
def test_sweep_output_matches_single_runs(tmp_path, backend):
    cfg = sweep_config(tmp_path / "sweep", backend)
    sweep = Sweep(cfg, {"laser.P0": [0.2, 0.3, 0.4]})
    with sweep.run() as result:
        manifest = json.loads((result.path / "sweep.json").read_text())
        assert manifest["parameter_keys"] == ["laser.P0"]
        assert manifest["parameters"] == sweep.members
        assert manifest["batch_sizes"] == [2, 1]
        assert manifest["members"][2] == {"group": 0, "batch": 1, "position": 0}
        assert result.parameter_table["laser.P0"].tolist() == [0.2, 0.3, 0.4]
        assert len(result) == 3
        assert len(list(result)) == 3
        assert result.fields["density"].shape == (5, 3, 16, 16)
        assert np.array_equal(result.fields["density"][:, 1], result.member(1).fields["density"][:])
        assert result.member(1).config["laser"]["P0"] == 0.3
        assert np.array_equal(result.scalars["norm"][:],
                              np.stack([result.member(i).scalars["norm"][:] for i in range(3)], axis=1))
        for index, member in enumerate(sweep.members):
            single_cfg = small_config(result__save_results=True, result__save_hdf5=True,
                                      result__save_psi=True,
                                      result__output_directory=str(tmp_path / f"single_{index}"),
                                      solver__total_time=0.04)
            single_cfg.laser.P0 = member["laser.P0"]
            controller = SimulationController(single_cfg)
            controller.run()
            with load_results(controller.run_directory) as single:
                view = result.member(index)
                for name in single.fields:
                    assert np.array_equal(view.fields[name][:], single.fields[name][:])
                for name in single.scalars:
                    assert np.array_equal(view.scalars[name][:], single.scalars[name][:])
                for name in single.static:
                    assert np.array_equal(view.static[name][:], single.static[name][:])
                for group in single.scalar_groups:
                    for label in single.scalar_groups[group]:
                        assert np.array_equal(view.scalar_groups[group][label][:],
                                              single.scalar_groups[group][label][:])
        if backend == "hdf5":
            with h5py.File(result.path / "batch_000" / "results.h5") as handle:
                assert handle.attrs["members"] == 2
                assert handle["member_index"][:].tolist() == [0, 1]
                assert handle["fields/density"].chunks[1] == 1
                assert handle["scalars/norm"].shape == (5, 2)
                assert handle["static/pump"].shape == (2, 16, 16)


def test_shared_static_field_and_member_view_validation(tmp_path):
    cfg = sweep_config(tmp_path)
    with Sweep(cfg, {"physics.g_C": [0.002, 0.003]}).run() as result:
        with h5py.File(result.path / "batch_000" / "results.h5") as handle:
            assert handle["static/pump"].shape == (16, 16)
        assert np.array_equal(result.member(0).static["pump"][:],
                              result.member(1).static["pump"][:])
    cfg.result.view_member = 2
    with pytest.raises(ValueError, match="view_member"):
        Sweep(cfg, {"physics.g_C": [0.002, 0.003]})
    with pytest.raises(ValueError, match="view_member"):
        SimulationController(cfg, members=[{}, {}])


def test_interleaved_groups_concatenate_in_member_order(tmp_path):
    cfg = sweep_config(tmp_path)
    parameters = {"boundary_condition.strength": [1.0, 2.0, 1.0],
                  "laser.P0": [0.2, 0.3, 0.4]}
    with Sweep(cfg, parameters, mode="zip").run() as result:
        assert result.manifest["batches"][0]["indices"] == [0, 2]
        assert result.manifest["batches"][1]["indices"] == [1]
        combined = result.fields["density"][:]
        for index in range(3):
            assert np.array_equal(combined[:, index], result.member(index).fields["density"][:])


def test_interrupted_sweep_resumes_bitwise(tmp_path, monkeypatch):
    cfg = sweep_config(tmp_path)
    cfg.checkpoint.every_steps = 2
    sweep = Sweep(cfg, {"laser.P0": [0.2, 0.3, 0.4]})
    with sweep.run() as complete:
        expected = {name: complete.fields[name][:].copy() for name in complete.fields}
    original = SimulationController._record_results

    def interrupt(self, t):
        original(self, t)
        if t >= 0.02:
            self._stop_requested = True

    monkeypatch.setattr(SimulationController, "_record_results", interrupt)
    assert sweep.run() is None
    monkeypatch.setattr(SimulationController, "_record_results", original)
    run_dir = sorted(tmp_path.iterdir())[-1]
    assert (run_dir / "batch_000" / "checkpoint.h5").exists()
    assert not (run_dir / "batch_000" / "complete").exists()
    cfg.sweep.batch_size = 1
    with sweep.run(resume=run_dir) as resumed:
        for name, value in expected.items():
            assert np.array_equal(resumed.fields[name][:], value)
        assert all((run_dir / f"batch_{i:03d}" / "complete").exists() for i in range(2))
        assert resumed.manifest["batch_sizes"] == [2, 1]


def test_json_sweep_member_reader(tmp_path):
    cfg = sweep_config(tmp_path, "json")
    with Sweep(cfg, {"laser.P0": [0.2, 0.3]}).run() as result:
        assert result.fields["density"].shape == (5, 2, 16, 16)
        assert result.member(1).scalars["norm"].shape == (5,)
        assert result.member(0).fields["psi"][:].dtype.kind == "c"


def test_checkpoint_rejects_changed_member_pump(tmp_path):
    cfg = sweep_config(tmp_path)
    cfg.checkpoint.every_steps = 2
    first = SimulationController(cfg, members=[{"laser.P0": 0.2}, {"laser.P0": 0.3}])
    first.run()
    second = SimulationController(cfg, members=[{"laser.P0": 0.2}, {"laser.P0": 0.4}])
    with pytest.raises(ValueError, match="member physics, pump"):
        second.load_checkpoint(first.run_directory / "checkpoint.h5")


def test_view_visitors_select_member():
    from polarism.results.result_node import ResultNode
    from polarism.results.visitors.animation_visitor import AnimationFieldSpec, AnimationVisitor
    from polarism.results.visitors.visualization_visitor import VisualizationVisitor

    class View:
        def update(self, **values):
            self.values = values

    field = ResultNode("density", lambda **ctx: None, lambda value: value,
                       "magma", None, None, True, True, None, is_field=True)
    scalar = ResultNode("norm", lambda **ctx: None, lambda value: value,
                        None, None, None, True, True, None, is_field=False)
    values = np.arange(8).reshape(2, 2, 2)
    view = View()
    VisualizationVisitor(view, member=1).visit_all(
        [field, scalar], cached={"density": (values, None), "norm": (np.array([1, 2]), np.array([1, 2]))},
        scalar_groups={"P_lasers": {"L0": np.array([3, 4])}}, t=1.0,
    )
    assert np.array_equal(view.values["fields"]["density"], values[1])
    assert view.values["scalars"]["norm"] == 2
    assert view.values["scalar_groups"]["P_lasers"]["L0"] == 4
    animation = AnimationVisitor("unused.mp4", [AnimationFieldSpec("density", "magma")], member=1)
    animation._encode_frame = lambda frames, t: setattr(animation, "selected", frames[0])
    animation.visit_all([field], cached={"density": (values, None)})
    assert np.array_equal(animation.selected, values[1])


def test_batched_scalar_observable_stays_per_member(tmp_path, monkeypatch):
    from polarism.compute_engine import compute_engine
    from polarism.results import ResultNode
    from polarism.results.observable_registry import available_observables

    def member_power(controller):
        return ResultNode("member_power", lambda **ctx: (compute_engine.xp.abs(ctx["state"].psi) ** 2).sum(axis=(-2, -1)),
                          lambda value: value, None, None, None, False, True, None,
                          is_field=False)

    monkeypatch.setitem(available_observables, "member-power", member_power)
    cfg = sweep_config(tmp_path)
    cfg.result.observables = ("member-power",)
    with Sweep(cfg, {"laser.P0": [0.2, 0.3]}).run() as result:
        assert result.scalars["member_power"].shape == (5, 2)
        for member in result:
            assert member.scalars["member_power"].shape == (5,)


@pytest.mark.gpu
def test_gpu_async_sweep_storage_matches_single_runs_and_avoids_main_thread_transfer(tmp_path, monkeypatch):
    from polarism.compute_engine import compute_engine, has_cuda_device

    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    members = [{"laser.P0": 0.2}, {"laser.P0": 0.3}]
    expected_fields, expected_scalars = {}, {}
    for index, overrides in enumerate(members):
        single = sweep_config(tmp_path / f"single_{index}")
        single.solver.precision = "single"
        single.compute_engine.use_gpu = True
        controller = SimulationController(single, members=[overrides])
        controller.run()
        with load_results(controller.run_directory) as result:
            for name in result.fields:
                expected_fields.setdefault(name, []).append(result.fields[name][:].copy())
            for name in result.scalars:
                expected_scalars.setdefault(name, []).append(result.scalars[name][:].copy())
    expected_fields = {name: np.stack(values, axis=1) for name, values in expected_fields.items()}
    expected_scalars = {name: np.stack(values, axis=1) for name, values in expected_scalars.items()}
    gpu = sweep_config(tmp_path / "gpu")
    gpu.solver.precision = "single"
    gpu.compute_engine.use_gpu = True
    controller = SimulationController(gpu, members=members)

    def unexpected_transfer(value):
        raise AssertionError("result frames transferred on the simulation thread")

    monkeypatch.setattr(compute_engine, "to_cpu", unexpected_transfer)
    controller.run()
    monkeypatch.undo()
    with load_results(controller.run_directory) as result:
        for name, expected in expected_fields.items():
            np.testing.assert_allclose(result.fields[name][:], expected, rtol=2e-5, atol=2e-5)
        for name, expected in expected_scalars.items():
            np.testing.assert_allclose(result.scalars[name][:], expected, rtol=2e-5, atol=2e-5)
