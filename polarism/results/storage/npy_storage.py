"""NumPy result storage."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from polarism.results.storage.base_storage import BaseStorage


class NPYStorage(BaseStorage):
    """Store result batches in NumPy files."""
    output_dir: Path
    batch_number: int

    def __init__(self, output_dir: Path, batch_size: int):
        """Set up NumPy batch storage."""
        super().__init__(output_dir, batch_size)
        self.batch_number = 0

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
        np.savez_compressed(batch_path, **arrays)

        self.batch_number += 1
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def finalize(self) -> None:
        """Flush remaining data and close NumPy storage."""
        if self.batch_count > 0:
            self.dump_batch()
