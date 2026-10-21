"""JSON result storage."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from polarism.results.storage.base_storage import BaseStorage
from polarism.results.storage.storage_registry import register_storage


@register_storage("json")
class JSONStorage(BaseStorage):
    """Store result batches in JSON files."""
    output_dir: Path
    batch_number: int

    def __init__(self, output_dir: Path, batch_size: int, config=None, x=None, y=None):
        """Set up JSON batch storage."""
        super().__init__(output_dir, batch_size)
        self.batch_number = 0
        self.members = getattr(config, "_storage_members", None)
        if getattr(config, "_storage_resume", False):
            self.batch_number = len(list(self.output_dir.glob("batch_*.json")))
        if self.members is not None:
            (self.output_dir / "member_index.json").write_text(json.dumps(config._storage_indices))
        if self.members is not None:
            from polarism import __version__
            self.metadata = {"format_version": 2, "polarism_version": __version__,
                             "config": asdict(config) if is_dataclass(config) else config or {},
                             "members": self.members}
            if x is not None:
                (self.output_dir / "grid_x.json").write_text(json.dumps(np.asarray(x).tolist()))
            if y is not None:
                (self.output_dir / "grid_y.json").write_text(json.dumps(np.asarray(y).tolist()))

    def dump_batch(self) -> None:
        """Write the current batch to JSON."""
        if not self.time_buffer:
            return

        batch_data = {
            "time": self.time_buffer[:],
            "fields": {},
            "scalars": {},
            "scalar_groups": {},
        }
        if self.batch_number == 0 and self.static_fields:
            batch_data["static"] = {
                name: self._json_field(field)
                for name, field in self.static_fields.items()
            }

        for field_name, field_list in self.field_buffers.items():
            batch_data["fields"][field_name] = [self._json_field(f) for f in field_list]

        for scalar_name, scalar_list in self.scalar_buffers.items():
            batch_data["scalars"][scalar_name] = np.asarray(scalar_list).tolist()

        for group_name, group_dict in self.scalar_group_buffers.items():
            batch_data["scalar_groups"][group_name] = {}
            for label, value_list in group_dict.items():
                batch_data["scalar_groups"][group_name][label] = np.asarray(value_list).tolist()

        batch_path = self.output_dir / f"batch_{self.batch_number:06d}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2)

        self.batch_number += 1
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def write_static(self, name: str, field: np.ndarray) -> None:
        """Keep a static field for the first JSON batch."""
        pass

    @staticmethod
    def _json_field(field: np.ndarray):
        """Encode real or complex arrays as JSON values."""
        if np.iscomplexobj(field):
            return {"real": field.real.tolist(), "imag": field.imag.tolist()}
        return field.tolist()

    def finalize(self) -> None:
        """Flush remaining data and close JSON storage."""
        if self.batch_count > 0:
            self.dump_batch()
        if self.members is not None:
            (self.output_dir / "metadata.json").write_text(json.dumps(self.metadata, sort_keys=True))
