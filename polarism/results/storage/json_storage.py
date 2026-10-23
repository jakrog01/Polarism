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
        self.ndim = config.grid.ndim if config is not None else (2 if y is not None else 1)
        self.batch_number = 0
        self._resume = getattr(config, "_storage_resume", False)
        metadata_path = self.output_dir / "metadata.json"
        if self._resume and metadata_path.exists():
            found = json.loads(metadata_path.read_text()).get("format_version")
            if found != 3:
                raise ValueError(f"results format v{found} is not supported by Polarism 0.7; supported: 3")
        self.members = getattr(config, "_storage_members", None)
        if self._resume:
            self.batch_number = len(list(self.output_dir.glob("batch_*.json")))
        if self.members is not None:
            member_path = self.output_dir / "member_index.json"
            if not self._resume or not member_path.exists():
                member_path.write_text(json.dumps(config._storage_indices))
        from polarism import __version__
        self.metadata = {"format_version": 3, "polarism_version": __version__,
                         "ndim": self.ndim, "datasets": {"time": {"label": "time", "units": "ps", "dims": ["time"]}},
                         "config": asdict(config) if is_dataclass(config) else config or {}}
        if self.members is not None:
            self.metadata["members"] = self.members
            self.metadata["datasets"]["member_index"] = {"dims": ["member"], "units": ""}
        if self.members is not None and x is not None:
            self.metadata["datasets"]["grid/x"] = {"units": "µm", "dims": ["x"]}
        if self.members is not None and y is not None:
            self.metadata["datasets"]["grid/y"] = {"units": "µm", "dims": ["y"]}
        if self.members is not None:
            if x is not None:
                grid_path = self.output_dir / "grid_x.json"
                if not self._resume or not grid_path.exists():
                    grid_path.write_text(json.dumps(np.asarray(x).tolist()))
            if y is not None:
                grid_path = self.output_dir / "grid_y.json"
                if not self._resume or not grid_path.exists():
                    grid_path.write_text(json.dumps(np.asarray(y).tolist()))

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
        for name in self.field_buffers:
            self.metadata["datasets"][f"fields/{name}"] = self.dataset_metadata(name, field=True)
        for name in self.scalar_buffers:
            self.metadata["datasets"][f"scalars/{name}"] = self.dataset_metadata(name)
        for group, values in self.scalar_group_buffers.items():
            for label in values:
                self.metadata["datasets"][f"scalar_groups/{group}/{label}"] = self.dataset_metadata(label)
        if self.batch_number == 1:
            (self.output_dir / "metadata.json").write_text(json.dumps(self.metadata, sort_keys=True))
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def write_static(self, name: str, field: np.ndarray) -> None:
        """Keep a static field for the first JSON batch."""
        metadata = self.dataset_metadata(name, static=True, field=True)
        if self.members is not None and field.ndim == self.ndim:
            metadata["dims"].remove("member")
        self.metadata["datasets"][f"static/{name}"] = metadata

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
        (self.output_dir / "metadata.json").write_text(json.dumps(self.metadata, sort_keys=True))
