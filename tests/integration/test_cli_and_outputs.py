from __future__ import annotations

import subprocess
import sys

import h5py
import matplotlib
import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.results.visitors import storage_visitor
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_cli_help() -> None:
    completed = subprocess.run([sys.executable, "-c", "from polarism.cli import main; main()", "--help"], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0
    assert any(token in completed.stdout for token in ("Usage:", "usage:", "--help"))


def test_programmatic_hdf5_output_and_figure(tmp_path) -> None:
    compute_engine.xp = np
    cfg = small_config(result__save_results=True, result__save_hdf5=True, result__output_directory=str(tmp_path), result__save_interval=1, result__batch_size=2)
    controller = SimulationController(cfg); controller.run()
    assert controller.storage_visitor is not None
    output = controller.storage_visitor.output_dir / "results.h5"
    with h5py.File(output) as result:
        time = result["time"][:]
        norm = result["scalars/N(t)"][:]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(time, norm); figure = tmp_path / "fig.png"; plt.savefig(figure); plt.close()
    assert figure.is_file() and figure.stat().st_size > 0


def test_controllers_use_unique_output_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(storage_visitor.time, "strftime", lambda _format: "20260101-120000")
    cfg = small_config(result__save_results=True, result__save_hdf5=True, result__output_directory=str(tmp_path))
    first = SimulationController(cfg)
    second = SimulationController(cfg)
    first.run()
    second.run()

    assert first.storage_visitor is not None
    assert second.storage_visitor is not None
    assert first.storage_visitor.output_dir == tmp_path / "20260101-120000"
    assert second.storage_visitor.output_dir == tmp_path / "20260101-120000-1"
    for controller in (first, second):
        with h5py.File(controller.storage_visitor.output_dir / "results.h5") as result:
            assert result["time"].shape[0] > 0


def test_fast_output_matches_uncompressed_cpu_run(tmp_path) -> None:
    import json

    from tests.unit.conftest import small_config

    paths = []
    for compression in ("none", "lzf"):
        cfg = small_config(
            result__save_results=True,
            result__save_hdf5=True,
            result__save_npy=True,
            result__save_json=True,
            result__save_psi=True,
            result__compression=compression,
            result__output_directory=str(tmp_path / compression),
            result__batch_size=3,
        )
        simulation = SimulationController(cfg)
        simulation.run()
        paths.append(simulation.storage_visitor.output_dir)
    with h5py.File(paths[0] / "results.h5") as plain, h5py.File(paths[1] / "results.h5") as fast:
        for name in ("time", "fields/|ψ|²", "fields/N_R", "fields/psi", "static/Pump", "scalars/N(t)"):
            assert np.array_equal(plain[name][:], fast[name][:])
        with np.load(paths[1] / "batch_000000.npz") as npy:
            assert np.array_equal(npy["field_psi"], fast["fields/psi"][:3])
            assert np.array_equal(np.load(paths[1] / "static_Pump.npy"), fast["static/Pump"][:])
        data = json.loads((paths[1] / "batch_000000.json").read_text())
        encoded = data["fields"]["psi"][0]
        assert np.array_equal(np.array(encoded["real"]) + 1j * np.array(encoded["imag"]), fast["fields/psi"][0])
        assert np.array_equal(np.array(data["static"]["Pump"]), fast["static/Pump"][:])


@pytest.mark.gpu
def test_gpu_fast_output_matches_synchronous_fields(tmp_path, monkeypatch) -> None:
    from polarism.compute_engine import compute_engine, has_cuda_device
    from polarism.results.visitors.storage_visitor import StorageVisitor

    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    cfg = small_config(
        solver__method="rk4-cuda",
        solver__precision="single",
        result__save_results=True,
        result__save_hdf5=True,
        result__save_psi=True,
        result__batch_size=2,
        result__output_directory=str(tmp_path / "async"),
    )
    cfg.compute_engine.use_gpu = True
    fast = SimulationController(cfg)
    fast.run()

    def synchronous_visit(self, nodes, t, **context):
        cached = {}
        selected = [node for node in nodes if node.save and node.name in context["cached"]]
        for node in selected:
            field = compute_engine.to_cpu(context["cached"][node.name][0]).copy()
            cached[node.name] = (field, None)
            if node.static:
                self._static_seen.add(node.name)
        groups = {
            name: {label: compute_engine.to_cpu(value) for label, value in values.items()}
            for name, values in context.get("scalar_groups", {}).items()
        }
        for storage in self.storages:
            storage.add_to_batch(t, selected, cached=cached, scalar_groups=groups)

    monkeypatch.setattr(StorageVisitor, "visit_all", synchronous_visit)
    cfg.result.output_directory = str(tmp_path / "sync")
    reference = SimulationController(cfg)
    reference.run()
    with h5py.File(fast.storage_visitor.output_dir / "results.h5") as actual, h5py.File(reference.storage_visitor.output_dir / "results.h5") as expected:
        for name in ("time", "fields/|ψ|²", "fields/N_R", "fields/psi", "static/Pump"):
            assert np.array_equal(actual[name][:], expected[name][:])


def test_cpu_fast_output_matches_synchronous_path(tmp_path, monkeypatch) -> None:
    from polarism.results.visitors.storage_visitor import StorageVisitor

    cfg = small_config(
        result__save_results=True,
        result__save_hdf5=True,
        result__save_npy=True,
        result__save_json=True,
        result__save_psi=True,
        result__batch_size=2,
        result__output_directory=str(tmp_path / "async"),
    )
    fast = SimulationController(cfg)
    fast.run()

    def synchronous_visit(self, nodes, t, **context):
        wanted = self.needs(nodes)
        selected = [node for node in nodes if node.name in wanted]
        cached = {
            node.name: (np.array(context["cached"][node.name][0], copy=True), None)
            for node in selected
        }
        self._static_seen.update(node.name for node in selected if node.static)
        groups = {
            name: {label: np.array(value, copy=True) for label, value in values.items()}
            for name, values in context.get("scalar_groups", {}).items()
        }
        for storage in self.storages:
            storage.add_to_batch(t, selected, cached=cached, scalar_groups=groups)

    monkeypatch.setattr(StorageVisitor, "visit_all", synchronous_visit)
    cfg.result.output_directory = str(tmp_path / "sync")
    reference = SimulationController(cfg)
    reference.run()
    with h5py.File(fast.storage_visitor.output_dir / "results.h5") as actual, h5py.File(reference.storage_visitor.output_dir / "results.h5") as expected:
        for group in ("fields", "static", "scalars", "scalar_groups"):
            actual_names = []
            actual[group].visititems(lambda name, obj: actual_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            for name in actual_names:
                assert np.array_equal(actual[f"{group}/{name}"][:], expected[f"{group}/{name}"][:])
        assert np.array_equal(actual["time"][:], expected["time"][:])
    with np.load(fast.storage_visitor.output_dir / "batch_000000.npz") as actual, np.load(reference.storage_visitor.output_dir / "batch_000000.npz") as expected:
        assert set(actual.files) == set(expected.files)
        for name in actual.files:
            assert np.array_equal(actual[name], expected[name])
    assert np.array_equal(
        np.load(fast.storage_visitor.output_dir / "static_Pump.npy"),
        np.load(reference.storage_visitor.output_dir / "static_Pump.npy"),
    )
    assert (fast.storage_visitor.output_dir / "batch_000000.json").read_text() == (
        reference.storage_visitor.output_dir / "batch_000000.json"
    ).read_text()
