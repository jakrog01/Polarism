"""Saved result reader compatibility and metadata."""
from __future__ import annotations

import h5py
import numpy as np
import pytest

import polarism as ps
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_hdf5_and_npy_round_trip_with_lazy_fields(tmp_path):
    cfg = small_config(result__save_results=True, result__save_hdf5=True,
                       result__save_npy=True, result__save_psi=True,
                       result__batch_size=2, result__output_directory=str(tmp_path))
    controller = SimulationController(cfg)
    controller.run()
    run_dir = controller.storage_visitor.output_dir
    with ps.load_results(run_dir) as result:
        assert set(("density", "psi", "N_R")) <= set(result.fields)
        assert "norm" in result.scalars
        assert "pump" in result.static
        assert np.array_equal(result.x, controller.grid.X[0, :])
        assert np.array_equal(result.y, controller.grid.Y[:, 0])
        assert result.attrs["format_version"] == 3
        assert result.config["result"]["save_psi"]
        psi = result.fields["psi"][:]
        assert np.allclose(result.fields["density"][:], psi.real ** 2 + psi.imag ** 2)
        assert np.array_equal(result.fields["density"][1], result.fields["density"][:][1])
        with h5py.File(run_dir / "results.h5") as handle:
            assert handle["fields/density"].attrs["label"] == "|ψ|²"
            assert handle["scalars/norm"].attrs["label"] == "N(t)"
            assert handle["static/pump"].attrs["label"] == "Pump"
            assert handle["time"].attrs["units"] == "ps"
            assert handle["grid/x"].attrs["units"] == "µm"
        hdf5_density = result.fields["density"][:]
        hdf5_time = result.time
    with pytest.raises((KeyError, ValueError)):
        result.fields["density"]
    (run_dir / "results.h5").unlink()
    with ps.load_results(run_dir) as result:
        assert np.array_equal(result.fields["density"], hdf5_density)
        assert np.array_equal(result.time, hdf5_time)
        assert "norm" in result.scalars
        assert "pump" in result.static
        assert "P_lasers" in result.scalar_groups
        assert result.attrs["format_version"] == 3


def test_v2_hdf5_is_rejected(tmp_path):
    path = tmp_path / "results.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = 2
        handle.create_dataset("time", data=[0.0, 1.0])
        handle.create_dataset("fields/|ψ|²", data=np.ones((2, 2, 3)))
        handle.create_dataset("scalars/N(t)", data=[2.0, 3.0])
        handle.create_dataset("static/Pump", data=np.ones((2, 3)))
    with pytest.raises(ValueError, match="results format v2.*supported: 3"):
        ps.load_results(path)


def test_old_animation_field_names_warn_and_select_new_nodes(tmp_path, monkeypatch):
    cfg = small_config(result__animation_fields=("|ψ|²", "Pump"))
    controller = SimulationController(cfg)
    class Recorder:
        def __init__(self, **kwargs):
            self.specs = kwargs["panel_specs"]

    monkeypatch.setattr("polarism.simulation_controller.AnimationVisitor", Recorder)
    with pytest.warns(DeprecationWarning) as warnings:
        controller._init_animation()
    assert len(warnings) == 2
    assert [(spec.source, spec.label) for spec in controller.animation_visitor.specs] == [
        ("density", "|ψ|²"), ("pump", "Pump")
    ]


def test_animation_renders_display_labels(tmp_path, monkeypatch):
    from polarism.results.visitors.animation_visitor import AnimationFieldSpec, AnimationVisitor

    class Encoder:
        def open(self, width, height, fps, path):
            self.size = (width, height)

    encoder = Encoder()
    monkeypatch.setattr("polarism.results.visitors.animation_visitor.create_encoder",
                        lambda backend, name: (encoder, "h264_nvenc"))
    visitor = AnimationVisitor(str(tmp_path / "labels.mp4"), [
        AnimationFieldSpec("density", "magma", label="|ψ|²"),
        AnimationFieldSpec("pump", "inferno", label="Pump"),
    ])
    visitor._vmins = [0.0, 0.0]
    visitor._vmaxs = [1.0, 1.0]
    visitor._open_encoder([np.ones((16, 128)), np.ones((16, 128))])
    assert encoder.size == (256, 48)
    assert visitor._title_strip.shape == (32, 256, 3)
    assert visitor._title_strip.max() > 0
