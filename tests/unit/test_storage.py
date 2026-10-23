from __future__ import annotations

import json

import h5py
import numpy as np

from polarism.results.result_node import ResultNode
from polarism.results.storage.hdf5_storage import HDF5Storage
from polarism.results.storage.json_storage import JSONStorage
from polarism.results.storage.npy_storage import NPYStorage

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


def _nodes():
    return [
        ResultNode("psi2", lambda **ctx: None, lambda x: x, "magma", None, None, True, True, None, True),
        ResultNode("N", lambda **ctx: None, lambda x: x, None, None, None, True, True, None, False),
    ]


def _frames(storage) -> None:
    nodes = _nodes()
    for t in (0.0, 0.1):
        storage.add_to_batch(t, nodes, cached={"psi2": (np.full((3, 4), t), 0.0), "N": (np.array(t), t + 1)}, scalar_groups={"group": {"label": t}})
    storage.finalize()


def test_batch_storage_layouts(tmp_path) -> None:
    hdf = HDF5Storage(tmp_path / "hdf", 2); _frames(hdf)
    with h5py.File(tmp_path / "hdf" / "results.h5") as f:
        assert f["time"].shape == (2,) and f["fields/psi2"].shape == (2, 1, 3, 4) and f["scalars/N"].shape == (2,)
        assert np.allclose(f["time"][:], [0.0, 0.1], rtol=TOL_MACHINE_F64)
    npy = NPYStorage(tmp_path / "npy", 2); _frames(npy)
    with np.load(tmp_path / "npy" / "batch_000000.npz") as data:
        assert {"time", "field_psi2", "scalar_N", "group_group_label"} <= set(data.files)
    js = JSONStorage(tmp_path / "json", 2); _frames(js)
    assert set(json.loads((tmp_path / "json" / "batch_000000.json").read_text())) == {"time", "fields", "scalars", "scalar_groups"}


def test_field_batches_flush_by_size(tmp_path, monkeypatch):
    from polarism import SimulationController, load_results
    from polarism.results.storage import base_storage
    from tests.unit.conftest import small_config

    monkeypatch.setattr(base_storage, "MAX_BATCH_BYTES", 1)
    cfg = small_config(result__save_results=True, result__save_npy=True, result__save_psi=True,
                       result__output_directory=str(tmp_path), solver__total_time=0.05)
    controller = SimulationController(cfg)
    controller.run()
    batches = sorted(controller.run_directory.glob("batch_*.npz"))
    with load_results(controller.run_directory) as result:
        assert len(batches) == len(result.time) == 6
        assert result.fields["psi"].shape == (6, 16, 16)
