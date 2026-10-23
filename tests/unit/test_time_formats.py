"""Fixed-time scheduling and versioned result and checkpoint files."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from polarism.config.validation import ConfigValidationError, validate_config
from polarism.results.reader import _ScalarComponentDataset, load_results
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_off_grid_start_and_checkpoint_resume(tmp_path):
    cfg = small_config(solver__total_time=0.03)
    run = SimulationController(cfg)
    run.set_initial_state(run.state.psi.copy(), t=0.0037)
    assert (run.state.t0, run.state.t, run.state.step) == (0.0037, 0.0037, 0)
    run.run()
    assert run.state.step == 3
    assert run.state.t == 0.0037 + 3 * cfg.solver.dt
    path = tmp_path / "checkpoint.h5"
    run.save_checkpoint(path)
    resumed = SimulationController(cfg)
    resumed.load_checkpoint(path)
    assert (resumed.state.t0, resumed.state.t, resumed.state.step) == (
        run.state.t0, run.state.t, run.state.step
    )


@pytest.mark.parametrize("field,value", [("result.save_every", 0.015), ("checkpoint.every", 0.015)])
def test_time_interval_nearest_values(field, value):
    cfg = small_config()
    section, name = field.split(".")
    setattr(getattr(cfg, section), name, value)
    with pytest.raises(ConfigValidationError, match="nearest valid values: 0.01 and 0.02 ps"):
        validate_config(cfg)


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("members", [None, [{}, {}]])
def test_results_v3_metadata(tmp_path, ndim, members):
    cfg = small_config(grid__ndim=ndim, solver__total_time=0.01,
                       result__save_results=True, result__save_hdf5=True,
                       result__save_npy=True, result__save_json=True,
                       result__output_directory=str(tmp_path))
    run = SimulationController(cfg, members=members)
    run.run()
    path = run.run_directory
    with h5py.File(path / "results.h5") as h5:
        assert h5.attrs["format_version"] == 3
        assert h5.attrs["ndim"] == ndim
        assert h5.attrs["grid_type"] == cfg.grid.grid_type
        assert list(h5.attrs["components"]) == ["scalar"]
        assert json.loads(h5.attrs["units"])["time"] == "ps"
        dims = ["time"] + (["member"] if members else []) + ["component"] + (["y", "x"] if ndim == 2 else ["x"])
        assert list(h5["fields/density"].attrs["dims"]) == dims
        assert h5["fields/density"].attrs["units"] == f"µm^-{ndim}"
        assert h5["time"].attrs["units"] == "ps"
        assert h5["grid/x"].attrs["units"] == "µm"
        assert list(h5["scalars/norm"].attrs["dims"]) == ["time"] + (["member"] if members else [])
        assert h5["scalars/norm"].attrs["units"] == ""
        assert list(h5["static/pump"].attrs["dims"])[-ndim:] == (["y", "x"] if ndim == 2 else ["x"])
        assert h5["static/pump"].attrs["units"] == ""
    metadata = json.loads((path / "metadata.json").read_text())
    assert metadata["format_version"] == 3
    assert metadata["ndim"] == ndim
    assert metadata["datasets"]["fields/density"]["dims"] == [
        dim for dim in dims if dim != "component"
    ]
    with load_results(path) as result:
        assert result.fields["density"].shape == (2, *([2] if members else []), *run.grid.shape)


@pytest.mark.parametrize("ndim,reservoir", [(1, "single"), (2, "double"), (2, "quadratic-double"), (1, "example-slow-single")])
def test_checkpoint_v2_reservoir_names(tmp_path, ndim, reservoir):
    if reservoir == "example-slow-single":
        from examples.plugins.polarism_plugins.reservoir import slow_single

    cfg = small_config(grid__ndim=ndim, reservoir__reservoir_type=reservoir)
    run = SimulationController(cfg)
    path = tmp_path / "checkpoint.h5"
    run.save_checkpoint(path)
    expected = {
        "single": {"nR"}, "double": {"nA", "nI"},
        "quadratic-double": {"nR", "nI"}, "example-slow-single": {"0"},
    }[reservoir]
    with h5py.File(path) as h5:
        assert h5.attrs["checkpoint_version"] == 2
        assert set(h5["reservoir"]) == expected
        assert h5["field"].shape == (1, 1, *run.grid.shape)
    restored = SimulationController(cfg)
    restored.load_checkpoint(path)
    np.testing.assert_array_equal(restored.state.field, run.state.field)


def test_old_versions_rejected(tmp_path):
    result_path = tmp_path / "results.h5"
    with h5py.File(result_path, "w") as h5:
        h5.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="results format v2.*supported: 3"):
        load_results(result_path)
    checkpoint_path = tmp_path / "checkpoint.h5"
    with h5py.File(checkpoint_path, "w") as h5:
        h5.attrs["checkpoint_version"] = 1
    with pytest.raises(ValueError, match="checkpoint format v1.*supported: 2"):
        SimulationController(small_config()).load_checkpoint(checkpoint_path)


@pytest.mark.parametrize("dims", [["time", "member", "component", "y", "x"],
                                  ["time", "component", "y", "x"]])
def test_scalar_field_slicing_matches_numpy(tmp_path, dims):
    shape = (5, 2, 1, 3, 4) if "member" in dims else (5, 1, 3, 4)
    data = np.arange(np.prod(shape)).reshape(shape)
    expected = np.squeeze(data, axis=dims.index("component"))
    with h5py.File(tmp_path / "field.h5", "w") as handle:
        dataset = handle.create_dataset("field", data=data)
        dataset.attrs["dims"] = dims
        field = _ScalarComponentDataset(dataset)
        assert field.shape == expected.shape
        assert np.array_equal(np.asarray(field), expected)
        for key in (np.s_[:], np.s_[...], np.s_[2], np.s_[1:3], np.s_[:, 1],
                    np.s_[..., 2], np.s_[0, ..., 1]):
            assert np.array_equal(field[key], expected[key])
