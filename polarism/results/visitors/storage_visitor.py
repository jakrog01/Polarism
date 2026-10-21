"""Result storage visitors."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.results.result_node import ResultNode
from polarism.results.storage import BaseStorage, available_storages
from polarism.results.visitors.result_visitor import ResultVisitor


def create_run_directory(root: str | Path) -> Path:
    """Create a unique timestamped run directory."""
    root = Path(root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        name = timestamp if suffix == 0 else f"{timestamp}-{suffix}"
        output_dir = root / name
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            return output_dir
        except FileExistsError:
            suffix += 1


class StorageVisitor(ResultVisitor):
    """Snapshot frames and write them on one bounded background worker."""
    data_location = "device"
    cfg: Config
    output_dir: Path
    batch_size: int
    storages: list[BaseStorage]

    def __init__(self, cfg: Config, x=None, y=None, output_dir=None):
        """Set up the storage backends for the config."""
        self.cfg = cfg
        self.output_dir = Path(output_dir) if output_dir is not None else create_run_directory(self.cfg.result.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = self.cfg.result.batch_size
        self.storages = []
        names = dict.fromkeys((
            *self.cfg.result.formats,
            *(name for enabled, name in (
                (self.cfg.result.save_hdf5, "hdf5"),
                (self.cfg.result.save_npy, "npy"),
                (self.cfg.result.save_json, "json"),
            ) if enabled),
        ))
        for name in names:
            self.storages.append(available_storages[name](self.output_dir, self.batch_size, self.cfg, x, y))
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._static_seen: set[str] = set()
        self._closed = False
        self._gpu = compute_engine.use_gpu
        self._device_id = cfg.compute_engine.gpu_device

    def needs(self, nodes: list[ResultNode]) -> set[str]:
        """Return names of nodes that are flagged for saving."""
        return {n.name for n in nodes if n.save and (not n.static or n.name not in self._static_seen)}

    def _start(self, frame_bytes: int) -> None:
        """Bound pending snapshots by frame count and device memory."""
        limit = 1024**3
        if self._gpu:
            import cupy as cp

            with cp.cuda.Device(self._device_id):
                limit = min(limit, cp.cuda.Device().mem_info[0] // 4)
        capacity = max(1, min(64, limit // max(frame_bytes, 1)))
        self._queue = queue.Queue(maxsize=capacity)
        self._thread = threading.Thread(target=self._write_loop, name="polarism-storage", daemon=True)
        self._thread.start()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Storage writer failed") from self._error

    def _put(self, item) -> None:
        assert self._queue is not None
        while True:
            self._raise_error()
            try:
                self._queue.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def visit_all(self, nodes: list[ResultNode], t: float, **context) -> None:
        """Copy mutable device results before the solver can advance."""
        self._raise_error()
        cached = context["cached"]
        needed = self.needs(nodes)
        selected = [n for n in nodes if n.name in needed and n.name in cached]
        snapshot = {}
        frame_bytes = 0
        for node in selected:
            value = cached[node.name][0]
            if getattr(self.cfg, "_storage_members", None) == 1 and not node.static:
                value = compute_engine.xp.expand_dims(value, axis=0)
            if self._gpu:
                import cupy as cp

                value = cp.array(value, copy=True)
            else:
                value = np.array(value, copy=True)
            snapshot[node.name] = (value, None)
            frame_bytes += value.nbytes
            if node.static:
                self._static_seen.add(node.name)
        groups = {}
        for group, values in context.get("scalar_groups", {}).items():
            groups[group] = {}
            for label, value in values.items():
                if getattr(self.cfg, "_storage_members", None) == 1:
                    value = compute_engine.xp.expand_dims(value, axis=0)
                copied = compute_engine.xp.array(value, copy=True)
                groups[group][label] = copied
                frame_bytes += copied.nbytes
        event = None
        if self._gpu:
            import cupy as cp

            event = cp.cuda.Event()
            event.record(cp.cuda.get_current_stream())
        if self._queue is None:
            self._start(frame_bytes)
        self._put((selected, t, snapshot, groups, event))

    def _write_loop(self) -> None:
        """Transfer ordered frames and feed every backend from one thread."""
        assert self._queue is not None
        pinned = {}
        stream = None
        try:
            if self._gpu:
                import cupy as cp

                cp.cuda.Device(self._device_id).use()
                stream = cp.cuda.Stream(non_blocking=True)
        except BaseException as exc:
            self._error = exc
            while True:
                item = self._queue.get()
                self._queue.task_done()
                if item is None:
                    return

        def host(value):
            if not self._gpu:
                return value
            key = (value.shape, value.dtype.str)
            if key not in pinned:
                try:
                    memory = cp.cuda.alloc_pinned_memory(value.nbytes)
                    array = np.frombuffer(memory, dtype=value.dtype).reshape(value.shape)
                    pinned[key] = (memory, array)
                except Exception:
                    pinned[key] = (None, None)
            array = pinned[key][1]
            if array is None:
                return value.get(stream=stream)
            value.get(out=array, stream=stream)
            return np.array(array, copy=True)

        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is not None:
                    continue
                nodes, t, snapshot, groups, event = item
                if event is not None:
                    stream.wait_event(event)
                cached = {name: (host(field), None) for name, (field, _) in snapshot.items()}
                host_groups = {
                    name: {label: host(value) for label, value in values.items()}
                    for name, values in groups.items()
                }
                for storage in self.storages:
                    storage.add_to_batch(t, nodes, cached=cached, scalar_groups=host_groups)
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def finalize(self) -> None:
        """Drain pending frames, join the worker, and close all backends."""
        if self._closed:
            self._raise_error()
            return
        self._closed = True
        if self._queue is not None:
            while True:
                try:
                    self._queue.put(None, timeout=0.05)
                    break
                except queue.Full:
                    continue
            self._thread.join()
        try:
            for storage in self.storages:
                storage.finalize()
        finally:
            self._raise_error()
