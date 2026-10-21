"""Fast result storage behaviour."""
from __future__ import annotations

import json
import threading
import time

import h5py
import numpy as np
import pytest

from polarism.config.simulation_parameters import Config
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.results.result_node import ResultNode
from polarism.results.storage.hdf5_storage import HDF5Storage
from polarism.results.storage.json_storage import JSONStorage
from polarism.results.storage.npy_storage import NPYStorage
from polarism.results.visitors.storage_visitor import StorageVisitor
from polarism.results.visitors.visualization_visitor import VisualizationVisitor


def _node(name, *, static=False, field=True):
    return ResultNode(name, lambda **ctx: None, lambda x: x, None, None, None, False, True, None, field, static)


def _config(tmp_path, **options):
    cfg = Config()
    cfg.result.output_directory = str(tmp_path)
    cfg.result.save_hdf5 = True
    cfg.result.batch_size = 3
    for name, value in options.items():
        setattr(cfg.result, name, value)
    return cfg


@pytest.mark.parametrize("compression,level,expected", [("none", 4, None), ("lzf", 4, "lzf"), ("gzip", 2, "gzip")])
def test_hdf5_compression_applies_to_all_datasets(tmp_path, compression, level, expected):
    cfg = _config(tmp_path, compression=compression, compression_level=level)
    storage = HDF5Storage(tmp_path, 2, cfg)
    nodes = [_node("field"), _node("static", static=True)]
    storage.add_to_batch(0.0, nodes, cached={"field": (np.ones((2, 3)), None), "static": (np.ones((2, 3)), None)})
    storage.finalize()
    with h5py.File(tmp_path / "results.h5") as result:
        for name in ("time", "fields/field", "static/static"):
            assert result[name].compression == expected
            assert result[name].shuffle == (expected is not None)
            if expected == "gzip":
                assert result[name].compression_opts == level


@pytest.mark.parametrize("compression", ["bad", "GZIP"])
def test_invalid_compression_rejected(compression):
    cfg = Config()
    cfg.result.compression = compression
    with pytest.raises(ConfigValidationError, match="compression"):
        validate_config(cfg)


def test_static_and_complex_round_trip_all_backends(tmp_path):
    nodes = [_node("psi"), _node("Pump", static=True)]
    psi = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64)
    pump = np.array([[4.0, 5.0]], dtype=np.float32)
    storages = [HDF5Storage(tmp_path / "h5", 2), NPYStorage(tmp_path / "npy", 2), JSONStorage(tmp_path / "json", 2)]
    for storage in storages:
        for t in (0.0, 1.0):
            storage.add_to_batch(t, nodes, cached={"psi": (psi, None), "Pump": (pump, None)})
        storage.finalize()
    with h5py.File(tmp_path / "h5" / "results.h5") as result:
        assert np.array_equal(result["fields/psi"][:], np.stack([psi, psi]))
        assert np.array_equal(result["static/Pump"][:], pump)
        assert "fields/Pump" not in result
    with np.load(tmp_path / "npy" / "batch_000000.npz") as result:
        assert np.array_equal(result["field_psi"], np.stack([psi, psi]))
        assert "field_Pump" not in result
    assert np.array_equal(np.load(tmp_path / "npy" / "static_Pump.npy"), pump)
    result = json.loads((tmp_path / "json" / "batch_000000.json").read_text())
    assert result["static"]["Pump"] == pump.tolist()
    encoded = result["fields"]["psi"][0]
    assert np.array_equal(np.array(encoded["real"]) + 1j * np.array(encoded["imag"]), psi)


def test_visitor_preserves_order_and_in_place_snapshot(tmp_path):
    visitor = StorageVisitor(_config(tmp_path))
    node = _node("density")
    shared = np.zeros((3, 4), dtype=np.float32)
    for index in range(31):
        shared.fill(index)
        visitor.visit_all([node], float(index), cached={"density": (shared, None)})
        shared.fill(-1)
    visitor.finalize()
    with h5py.File(visitor.output_dir / "results.h5") as result:
        assert np.array_equal(result["time"][:], np.arange(31, dtype=float))
        assert np.array_equal(result["fields/density"][:, 0, 0], np.arange(31, dtype=np.float32))


def test_writer_failure_reaches_main_thread(tmp_path):
    visitor = StorageVisitor(_config(tmp_path))
    class Broken:
        def add_to_batch(self, *args, **kwargs):
            raise OSError("disk failure")

        def finalize(self):
            pass

    visitor.storages = [Broken()]
    node = _node("field")
    visitor.visit_all([node], 0.0, cached={"field": (np.ones((2, 2)), None)})
    with pytest.raises(RuntimeError, match="Storage writer failed") as error:
        visitor.finalize()
    assert isinstance(error.value.__cause__, OSError)


def test_bounded_queue_blocks_producer(tmp_path):
    visitor = StorageVisitor(_config(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    class Slow:
        def add_to_batch(self, *args, **kwargs):
            entered.set()
            release.wait(timeout=5)

        def finalize(self):
            pass

    visitor.storages = [Slow()]
    node = _node("field")
    visitor.visit_all([node], 0.0, cached={"field": (np.ones((2, 2)), None)})
    assert entered.wait(timeout=2)
    visitor._queue.maxsize = 1
    visitor.visit_all([node], 1.0, cached={"field": (np.ones((2, 2)), None)})
    done = threading.Event()
    def produce():
        visitor.visit_all([node], 2.0, cached={"field": (np.ones((2, 2)), None)})
        done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    time.sleep(0.05)
    assert not done.is_set()
    assert visitor._queue.qsize() == 1
    release.set()
    producer.join(timeout=2)
    assert done.is_set()
    visitor.finalize()


def test_continuous_pump_static_flag_respects_delay():
    from polarism.simulation_controller import SimulationController
    from tests.unit.conftest import small_config

    for laser_type in ("continuous-gaussian", "continuous-exp", "continuous-exp-length", "uniform"):
        cfg = small_config(laser__laser_type=laser_type)
        controller = SimulationController(cfg)
        pump = next(node for node in controller._build_result_nodes() if node.name == "Pump")
        assert pump.static
        assert all(laser.power_coefficient(0.0) == laser.power_coefficient(1.0) for laser in controller.lasers)
        cfg.laser.delay = 0.1
        controller = SimulationController(cfg)
        pump = next(node for node in controller._build_result_nodes() if node.name == "Pump")
        assert not pump.static
    cfg = small_config(laser__laser_type="pulse-gaussian")
    controller = SimulationController(cfg)
    pump = next(node for node in controller._build_result_nodes() if node.name == "Pump")
    assert not pump.static


def test_result_options_accept_yaml_mapping():
    import yaml

    from polarism.config.simulation_parameters import ResultParameters

    options = yaml.safe_load("compression: gzip\ncompression_level: 2\nsave_psi: true\n")
    result = ResultParameters(**options)
    assert result.compression == "gzip"
    assert result.compression_level == 2
    assert result.save_psi


def test_visualization_receives_python_floats_for_backend_scalars():
    class Recorder:
        def update(self, **kwargs):
            self.kwargs = kwargs

    recorder = Recorder()
    node = ResultNode("N", lambda **ctx: np.float32(2.0), lambda v: v, None, None, None, True, True, None, False)
    VisualizationVisitor(recorder).visit_all(
        [node], t=0.0, cached={"N": (np.array(2.0), np.array(2.0))},
        scalar_groups={"P": {"L0": np.array(3.0)}},
    )
    assert type(recorder.kwargs["scalars"]["N"]) is float
    assert type(recorder.kwargs["scalar_groups"]["P"]["L0"]) is float
