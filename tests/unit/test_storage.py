from __future__ import annotations

import json
import tracemalloc

import h5py
import numpy as np

from polarism.compute_engine import compute_engine
from polarism.results.result_node import ResultNode
from polarism.results.storage.appendable_hdf5 import CpuBufferedHDF5Writer, create_hdf5_writer
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
        assert f["time"].shape == (2,) and f["fields/psi2"].shape == (2, 3, 4) and f["scalars/N"].shape == (2,)
        assert np.allclose(f["time"][:], [0.0, 0.1], rtol=TOL_MACHINE_F64)
    npy = NPYStorage(tmp_path / "npy", 2); _frames(npy)
    with np.load(tmp_path / "npy" / "batch_000000.npz") as data:
        assert {"time", "field_psi2", "scalar_N", "group_group_label"} <= set(data.files)
    js = JSONStorage(tmp_path / "json", 2); _frames(js)
    assert set(json.loads((tmp_path / "json" / "batch_000000.json").read_text())) == {"time", "fields", "scalars", "scalar_groups"}


def test_appendable_cpu_writer_is_cupy_independent(tmp_path) -> None:
    compute_engine.use_gpu = False; compute_engine.xp = np
    writer = create_hdf5_writer(str(tmp_path / "out.h5"), 4, {"psi": np.complex128, "rho": np.float64}, (16, 16))
    assert isinstance(writer, CpuBufferedHDF5Writer)
    writer.register_scalar("N")
    frames = []
    for index in range(10):
        psi = np.full((16, 16), index + 1j * index, dtype=np.complex128); frames.append(psi)
        writer.record(index * 0.1, {"psi": psi, "rho": psi.real}, {"N": float(index)}, mode=0)
    writer.close()
    with h5py.File(tmp_path / "out.h5") as f:
        assert f["time"].shape == (10,) and f["fields/psi"].dtype == np.dtype(np.complex128)
        assert np.array_equal(f["fields/psi"][:], np.stack(frames))


def test_cpu_buffered_writer_memory_is_bounded_by_batch_size(tmp_path) -> None:
    """Verify that total trajectory length does not enlarge the frame buffer.

    The allowance of two batches covers the active batch, the contiguous array
    assembled during a flush, h5py metadata, and allocator variation.
    """
    compute_engine.use_gpu = False
    compute_engine.xp = np
    batch_size = 4
    shape = (128, 128)
    frame_bytes = int(np.prod(shape) * np.dtype(np.complex128).itemsize)

    def measure(frame_count: int, name: str) -> tuple[int, int]:
        tracemalloc.start()
        writer = CpuBufferedHDF5Writer(str(tmp_path / name), batch_size)
        maximum_buffered = 0
        for index in range(frame_count):
            frame = np.full(shape, index + 1j * index, dtype=np.complex128)
            writer.record(float(index), {"psi": frame}, {})
            maximum_buffered = max(
                maximum_buffered,
                len(writer._field_bufs.get("psi", [])),
            )
        writer.close()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return maximum_buffered, peak

    short_buffered, short_peak = measure(2 * batch_size, "short.h5")
    long_buffered, long_peak = measure(32 * batch_size, "long.h5")
    assert short_buffered <= batch_size - 1
    assert long_buffered <= batch_size - 1
    assert long_peak - short_peak < 2 * batch_size * frame_bytes
