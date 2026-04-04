"""Result storage visitors."""
from __future__ import annotations

import time
from pathlib import Path

from polarism.config.simulation_parameters import Config
from polarism.results.result_node import ResultNode
from polarism.results.storage.hdf5_storage import HDF5Storage
from polarism.results.storage.json_storage import JSONStorage
from polarism.results.storage.npy_storage import NPYStorage
from polarism.results.visitors.result_visitor import ResultVisitor


class StorageVisitor(ResultVisitor):
    """Write result nodes to storage backends."""
    cfg: Config
    output_dir: Path
    batch_size: int
    storages: list[HDF5Storage | JSONStorage | NPYStorage]

    def __init__(self, cfg: Config):
        """Set up the storage backends for the config."""
        self.cfg = cfg
        self.output_dir = Path(
            self.cfg.result.output_directory, time.strftime("%Y%m%d-%H%M%S")
        )
        self.batch_size = self.cfg.result.batch_size

        self.storages = []

        if self.cfg.result.save_hdf5:
            self.storages.append(HDF5Storage(self.output_dir, self.batch_size))

        if self.cfg.result.save_json:
            self.storages.append(JSONStorage(self.output_dir, self.batch_size))

        if self.cfg.result.save_npy:
            self.storages.append(NPYStorage(self.output_dir, self.batch_size))

    def visit_all(self, nodes: list[ResultNode], t: float, **context) -> None:
        """Store one result step."""
        for storage in self.storages:
            storage.add_to_batch(t, nodes, **context)

    def finalize(self) -> None:
        """Flush all storage backends."""
        for storage in self.storages:
            storage.finalize()
