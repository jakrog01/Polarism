"""NumPy result storage."""
from __future__ import annotations

from pathlib import Path
import json
from dataclasses import asdict, is_dataclass

import numpy as np

from polarism.results.storage.base_storage import BaseStorage
from polarism.results.storage.storage_registry import register_storage


@register_storage("npy")
class NPYStorage(BaseStorage):
    """Store result batches in NumPy files."""
    output_dir: Path
    batch_number: int

    def __init__(self, output_dir: Path, batch_size: int, config=None, x=None, y=None,
                 compress: bool | None = None):
        """Set up NumPy batch storage."""
        super().__init__(output_dir, batch_size)
        self.batch_number = 0
        self._resume = getattr(config, "_storage_resume", False)
        self.members = getattr(config, "_storage_members", None)
        if self._resume:
            self.batch_number = len(list(self.output_dir.glob("batch_*.npz")))
        if isinstance(config, bool):
            compress = config if compress is None else compress
            config = None
        self.compress = (config.result.compression != "none") if compress is None and config is not None else bool(compress)
        from polarism import __version__
        self.metadata = {
            "format_version": 2,
            "polarism_version": __version__,
            "datasets": {"time": {"label": "time", "units": "ps"}},
            "config": asdict(config) if is_dataclass(config) else config or {},
        }
        if self.members is not None:
            self.metadata["members"] = self.members
            member_path = self.output_dir / "member_index.npy"
            if not self._resume or not member_path.exists():
                np.save(member_path, np.asarray(config._storage_indices))
        if x is not None:
            grid_path = self.output_dir / "grid_x.npy"
            if not self._resume or not grid_path.exists():
                np.save(grid_path, np.asarray(x))
            self.metadata["datasets"]["grid/x"] = {"units": "µm"}
        if y is not None:
            grid_path = self.output_dir / "grid_y.npy"
            if not self._resume or not grid_path.exists():
                np.save(grid_path, np.asarray(y))
            self.metadata["datasets"]["grid/y"] = {"units": "µm"}

    def dump_batch(self) -> None:
        """Write the current batch to NumPy files."""
        if not self.time_buffer:
            return

        arrays = {}
        arrays["time"] = np.array(self.time_buffer)

        for field_name, field_list in self.field_buffers.items():
            arrays[f"field_{field_name}"] = np.stack(field_list, axis=0)

        for scalar_name, scalar_list in self.scalar_buffers.items():
            arrays[f"scalar_{scalar_name}"] = np.array(scalar_list)

        for group_name, group_dict in self.scalar_group_buffers.items():
            for label, value_list in group_dict.items():
                key = f"group_{group_name}_{label}"
                arrays[key] = np.array(value_list)

        batch_path = self.output_dir / f"batch_{self.batch_number:06d}.npz"
        (np.savez_compressed if self.compress else np.savez)(batch_path, **arrays)

        self.batch_number += 1
        for name in self.field_buffers:
            self.metadata["datasets"][f"fields/{name}"] = self.node_metadata[name]
        for name in self.scalar_buffers:
            self.metadata["datasets"][f"scalars/{name}"] = self.node_metadata[name]
        for group, values in self.scalar_group_buffers.items():
            for label in values:
                self.metadata["datasets"][f"scalar_groups/{group}/{label}"] = {"label": label}
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def write_static(self, name: str, field: np.ndarray) -> None:
        """Store a field once as a NumPy file."""
        path = self.output_dir / f"static_{name}.npy"
        if not self._resume or not path.exists():
            np.save(path, field)
        self.metadata["datasets"][f"static/{name}"] = self.node_metadata.get(name, {"label": name})

    def finalize(self) -> None:
        """Flush remaining data and close NumPy storage."""
        if self.batch_count > 0:
            self.dump_batch()
        path = self.output_dir / "metadata.json"
        if path.exists() and self.batch_number > 0:
            previous = json.loads(path.read_text())
            previous["datasets"].update(self.metadata["datasets"])
            self.metadata = previous
        path.write_text(json.dumps(self.metadata, sort_keys=True))
