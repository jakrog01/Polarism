"""JSON result storage."""
from __future__ import annotations

import json
from pathlib import Path

from polarism.results.storage.base_storage import BaseStorage


class JSONStorage(BaseStorage):
    """Store result batches in JSON files."""
    output_dir: Path
    batch_number: int

    def __init__(self, output_dir: Path, batch_size: int):
        """Set up JSON batch storage."""
        super().__init__(output_dir, batch_size)
        self.batch_number = 0

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

        for field_name, field_list in self.field_buffers.items():
            batch_data["fields"][field_name] = [f.tolist() for f in field_list]

        for scalar_name, scalar_list in self.scalar_buffers.items():
            batch_data["scalars"][scalar_name] = scalar_list[:]

        for group_name, group_dict in self.scalar_group_buffers.items():
            batch_data["scalar_groups"][group_name] = {}
            for label, value_list in group_dict.items():
                batch_data["scalar_groups"][group_name][label] = value_list[:]

        batch_path = self.output_dir / f"batch_{self.batch_number:06d}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2)

        self.batch_number += 1
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def finalize(self) -> None:
        """Flush remaining data and close JSON storage."""
        if self.batch_count > 0:
            self.dump_batch()
