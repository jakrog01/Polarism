"""Appendable HDF5 writer backends.

Provides a CPU-buffered backend and a GPU-async backend (double-buffered,
pinned host memory, non-blocking CuPy stream) behind a common abstract
interface, plus a factory function and a GPU-memory-aware batch-size heuristic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import h5py
import numpy as np

from polarism.compute_engine import compute_engine

_RESERVED_GPU_GB = 4.0


class AppendableHDF5Writer(ABC):
    """Abstract append-only HDF5 writer."""

    @abstractmethod
    def register_scalar(self, name: str) -> None:
        """Register a named scalar series before the first :meth:`record` call."""

    @abstractmethod
    def record(
        self,
        t: float,
        fields: dict[str, Any],
        scalars: dict[str, float],
        mode: int = 0,
    ) -> None:
        """Append one time frame to the writer."""

    @abstractmethod
    def close(self) -> None:
        """Flush all pending data and close the HDF5 file."""

    @property
    @abstractmethod
    def total(self) -> int:
        """Total number of frames written so far."""


class CpuBufferedHDF5Writer(AppendableHDF5Writer):
    """HDF5 writer that accumulates frames in Python lists and flushes in batches.

    Uses ``lzf`` compression.  All device arrays are transferred to host via
    :func:`~polarism.compute_engine.ComputeEngine.to_cpu` before buffering.
    """

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        flush_every: int = 10,
    ) -> None:
        """Open the output file and prepare internal buffers.

        Parameters
        ----------
        filepath : str
            Path of the HDF5 file to create.
        batch_size : int
            Number of frames to accumulate before flushing to disk.
        flush_every : int
            Number of batch flushes between explicit ``h5py`` file flushes.
        """
        self.batch_size = batch_size
        self._flush_every = flush_every
        self._h5 = h5py.File(filepath, "w")
        self._datasets: dict[str, h5py.Dataset] = {}
        self._total = 0

        self._time_buf: list[float] = []
        self._mode_buf: list[int] = []
        self._field_bufs: dict[str, list[np.ndarray]] = {}
        self._scalar_bufs: dict[str, list[float]] = {}
        self._count = 0
        self._writes_since_flush = 0

    def register_scalar(self, name: str) -> None:
        self._scalar_bufs[name] = []

    def record(
        self,
        t: float,
        fields: dict[str, Any],
        scalars: dict[str, float],
        mode: int = 0,
    ) -> None:
        self._time_buf.append(t)
        self._mode_buf.append(mode)
        for nm, data in fields.items():
            if nm not in self._field_bufs:
                self._field_bufs[nm] = []
            self._field_bufs[nm].append(compute_engine.to_cpu(data))
        for nm, val in scalars.items():
            self._scalar_bufs[nm].append(float(val))
        self._count += 1
        if self._count >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        n = self._count
        if n == 0:
            return

        self._append("time", np.array(self._time_buf[:n]), scalar=True)
        self._append("mode", np.array(self._mode_buf[:n], dtype=np.int8), scalar=True)
        for nm, buf in self._field_bufs.items():
            self._append(f"fields/{nm}", np.array(buf[:n]), scalar=False)
        for nm, buf in self._scalar_bufs.items():
            self._append(f"scalars/{nm}", np.array(buf[:n]), scalar=True)

        self._total += n
        self._count = 0
        self._time_buf.clear()
        self._mode_buf.clear()
        self._field_bufs.clear()
        for k in self._scalar_bufs:
            self._scalar_bufs[k] = []

        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self._h5.flush()
            self._writes_since_flush = 0

    def _append(self, name: str, data: np.ndarray, scalar: bool) -> None:
        if name not in self._datasets:
            if scalar:
                self._datasets[name] = self._h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None,),
                    compression="lzf",
                    chunks=(min(1000, self.batch_size),),
                )
            else:
                self._datasets[name] = self._h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *data.shape[1:]),
                    compression="lzf",
                    chunks=(min(32, self.batch_size), *data.shape[1:]),
                )
        else:
            ds = self._datasets[name]
            old = ds.shape[0]
            ds.resize(old + data.shape[0], axis=0)
            ds[old:] = data

    def close(self) -> None:
        self._flush_batch()
        self._h5.close()

    @property
    def total(self) -> int:
        return self._total


class GpuAsyncHDF5Writer(AppendableHDF5Writer):
    """HDF5 writer with double-buffered GPU staging and non-blocking transfers.

    Uses pinned host memory when CUDA host registration succeeds, falling back
    to ordinary host arrays otherwise.  All heavy device-to-host copies overlap
    with the next simulation batch via a non-blocking CuPy stream.

    Uses ``lzf`` compression, matching the compression policy of the original
    ``AsyncBatchWriter``.

    Parameters
    ----------
    filepath : str
        Path of the HDF5 file to create.
    batch_size : int
        Number of frames per double-buffer slot.
    field_specs : dict[str, np.dtype]
        Mapping of field name to dtype.  Complex dtypes allocate complex128
        device buffers; all others allocate float64 device buffers.
    spatial_shape : tuple[int, int]
        ``(ny, nx)`` — the 2-D shape of each field frame.
    flush_every : int
        Number of batch writes between explicit ``h5py`` file flushes.
    """

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        field_specs: dict[str, np.dtype],
        spatial_shape: tuple[int, int],
        flush_every: int = 10,
    ) -> None:
        self.xp = compute_engine.xp
        self.batch_size = batch_size
        self._spatial_shape = spatial_shape
        self._flush_every = flush_every

        ny, nx = spatial_shape
        self._h5 = h5py.File(filepath, "w")
        self._datasets: dict[str, h5py.Dataset] = {}
        self._total = 0

        self._field_names = list(field_specs.keys())
        self._field_dtypes = {
            nm: (self.xp.complex128 if np.issubdtype(dt, np.complexfloating) else self.xp.float64)
            for nm, dt in field_specs.items()
        }
        self._host_dtypes = {
            nm: (np.complex128 if np.issubdtype(dt, np.complexfloating) else np.float64)
            for nm, dt in field_specs.items()
        }

        self._gpu: list[dict[str, Any]] = [{}, {}]
        for i in range(2):
            for nm, dtype in self._field_dtypes.items():
                self._gpu[i][nm] = self.xp.zeros((batch_size, ny, nx), dtype=dtype)

        self._host: dict[str, np.ndarray] = {}
        self._pinned = False
        try:
            import cupy as cp

            for nm, dtype in self._host_dtypes.items():
                buf = np.empty((batch_size, ny, nx), dtype=dtype)
                cp.cuda.runtime.hostRegister(buf.ctypes.data, buf.nbytes, 0)
                self._host[nm] = buf
            self._transfer_stream = cp.cuda.Stream(non_blocking=True)
            self._event = cp.cuda.Event()
            self._pinned = True
        except Exception:
            self._host = {}
            self._pinned = False

        if not self._pinned:
            for nm, dtype in self._host_dtypes.items():
                self._host[nm] = np.empty((batch_size, ny, nx), dtype=dtype)

        self._active = 0
        self._count = 0
        self._transfer_pending = False
        self._pending_n = 0
        self._time_buf: list[float] = []
        self._mode_buf: list[int] = []
        self._scalar_bufs: dict[str, list[float]] = {}
        self._pend_time: list[float] | None = None
        self._pend_mode: list[int] | None = None
        self._pend_scalars: dict[str, list[float]] | None = None
        self._writes_since_flush = 0

    def register_scalar(self, name: str) -> None:
        self._scalar_bufs[name] = []

    def record(
        self,
        t: float,
        fields: dict[str, Any],
        scalars: dict[str, float],
        mode: int = 0,
    ) -> None:
        if self._transfer_pending:
            self._wait_and_write()

        idx = self._count
        buf = self._gpu[self._active]

        self._time_buf.append(t)
        self._mode_buf.append(mode)
        for nm, data in fields.items():
            buf[nm][idx] = data
        for nm, val in scalars.items():
            self._scalar_bufs[nm].append(float(val))

        self._count += 1
        if self._count >= self.batch_size:
            self._start_transfer()

    def _start_transfer(self) -> None:
        n = self._count
        buf = self._gpu[self._active]

        if self._pinned:
            for nm in self._field_names:
                buf[nm][:n].get(out=self._host[nm][:n], stream=self._transfer_stream)
            self._event.record(self._transfer_stream)
        else:
            for nm in self._field_names:
                self._host[nm][:n] = compute_engine.to_cpu(buf[nm][:n])

        self._pend_time = list(self._time_buf)
        self._pend_mode = list(self._mode_buf)
        self._pend_scalars = {k: list(v) for k, v in self._scalar_bufs.items()}
        self._pending_n = n
        self._transfer_pending = True

        self._active = 1 - self._active
        self._count = 0
        self._time_buf = []
        self._mode_buf = []
        for k in self._scalar_bufs:
            self._scalar_bufs[k] = []

    def _wait_and_write(self) -> None:
        if self._pinned:
            self._event.synchronize()

        n = self._pending_n
        self._append("time", np.array(self._pend_time[:n]), scalar=True)
        self._append("mode", np.array(self._pend_mode[:n], dtype=np.int8), scalar=True)

        for nm in self._field_names:
            self._append(f"fields/{nm}", self._host[nm][:n], scalar=False)

        for nm, vals in self._pend_scalars.items():
            self._append(f"scalars/{nm}", np.array(vals[:n]), scalar=True)

        self._total += n
        self._transfer_pending = False
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self._h5.flush()
            self._writes_since_flush = 0

    def _append(self, name: str, data: np.ndarray, scalar: bool) -> None:
        if name not in self._datasets:
            if scalar:
                self._datasets[name] = self._h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None,),
                    compression="lzf",
                    chunks=(min(1000, self.batch_size),),
                )
            else:
                self._datasets[name] = self._h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *data.shape[1:]),
                    compression="lzf",
                    chunks=(min(32, self.batch_size), *data.shape[1:]),
                )
        else:
            ds = self._datasets[name]
            old = ds.shape[0]
            ds.resize(old + data.shape[0], axis=0)
            ds[old:] = data

    def close(self) -> None:
        if self._transfer_pending:
            self._wait_and_write()
        if self._count > 0:
            self._start_transfer()
            self._wait_and_write()
        if self._pinned:
            import cupy as cp

            for buf in self._host.values():
                try:
                    cp.cuda.runtime.hostUnregister(buf.ctypes.data)
                except Exception:
                    pass
        self._h5.close()

    @property
    def total(self) -> int:
        return self._total


def compute_batch_size(
    field_specs: dict[str, np.dtype],
    spatial_shape: tuple[int, int],
    reserved_gpu_gb: float = _RESERVED_GPU_GB,
) -> int:
    """Estimate a safe HDF5 write-buffer depth for the current GPU.

    Allocates 75 % of free GPU memory (after reserving ``reserved_gpu_gb`` GB)
    across both double-buffer GPU slots.  Returns 1 when memory is too tight to
    fit even one frame per slot — correct but slower, as every :meth:`record`
    call triggers an immediate transfer.  Returns 500 on the CPU path where
    host memory is not the limiting resource.

    Parameters
    ----------
    field_specs : dict[str, np.dtype]
        Mapping of field name to dtype — the same dict passed to
        :func:`create_hdf5_writer`.
    spatial_shape : tuple[int, int]
        ``(ny, nx)`` spatial grid dimensions.
    reserved_gpu_gb : float
        GPU memory (in GiB) to leave untouched for the solver.

    Returns
    -------
    int
        Batch depth in frames, clipped to ``[1, 10000]``.
    """
    xp = compute_engine.xp
    if not hasattr(xp, "cuda"):
        return 500

    ny, nx = spatial_shape
    free = xp.cuda.Device().mem_info[0]
    usable = max(0, free - int(reserved_gpu_gb * 1024**3))
    frame_bytes = sum(np.dtype(dt).itemsize * ny * nx for dt in field_specs.values())
    if frame_bytes == 0:
        return 500
    batch = int(usable * 0.75) // (2 * frame_bytes)
    return max(1, min(batch, 10000))


def create_hdf5_writer(
    filepath: str,
    batch_size: int,
    field_specs: dict[str, np.dtype],
    spatial_shape: tuple[int, int],
    flush_every: int = 10,
) -> AppendableHDF5Writer:
    """Return the appropriate appendable HDF5 writer for the active compute backend.

    Selects :class:`GpuAsyncHDF5Writer` when ``compute_engine.use_gpu`` is
    ``True``, otherwise :class:`CpuBufferedHDF5Writer`.

    Parameters
    ----------
    filepath : str
        Path of the HDF5 file to create.
    batch_size : int
        Number of frames per write batch.
    field_specs : dict[str, np.dtype]
        Mapping of field name to dtype.  Used by the GPU backend for device
        buffer pre-allocation; the CPU backend infers types on the fly.
    spatial_shape : tuple[int, int]
        ``(ny, nx)`` — used by the GPU backend for buffer pre-allocation.
    flush_every : int
        Batches between explicit ``h5py`` file flushes.

    Returns
    -------
    AppendableHDF5Writer
        Concrete writer instance ready for :meth:`~AppendableHDF5Writer.register_scalar`
        and :meth:`~AppendableHDF5Writer.record` calls.
    """
    if compute_engine.use_gpu:
        return GpuAsyncHDF5Writer(filepath, batch_size, field_specs, spatial_shape, flush_every)

    return CpuBufferedHDF5Writer(filepath, batch_size, flush_every)
