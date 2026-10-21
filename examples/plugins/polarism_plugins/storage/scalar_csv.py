"""Small CSV backend for scalar result nodes."""
from __future__ import annotations

import csv

from polarism.results.storage import BaseStorage, register_storage


@register_storage("example-scalar-csv")
class ScalarCSVStorage(BaseStorage):
    """Write scalar nodes to a single CSV file."""

    def __init__(self, output_dir, batch_size, config, x, y):
        """Set up scalar CSV storage."""
        super().__init__(output_dir, batch_size)
        self.path = output_dir / "scalars.csv"
        self.columns = None

    def add_to_batch(self, t, nodes, **context):
        """Keep only scalar nodes for the shared batch implementation."""
        scalars = [node for node in nodes if node.save and not node.is_field]
        super().add_to_batch(t, scalars, **context)

    def dump_batch(self):
        """Append buffered scalar rows."""
        if not self.time_buffer:
            return
        columns = list(self.scalar_buffers)
        if self.columns is None:
            self.columns = columns
        with self.path.open("a", newline="") as stream:
            writer = csv.writer(stream)
            if self.total_saved == 0:
                writer.writerow(["time", *self.columns])
            for index, t in enumerate(self.time_buffer):
                writer.writerow([t, *(self.scalar_buffers[name][index] for name in self.columns)])
        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def write_static(self, name, field):
        """Ignore static fields in this scalar-only format."""

    def finalize(self):
        """Flush the remaining rows."""
        self.dump_batch()
