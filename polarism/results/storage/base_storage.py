from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseStorage(ABC):
    output_dir: Path
    batch_size: int
    batch_count: int
    total_saved: int

    time_buffer: list[float]
    field_buffers: dict[str, list[np.ndarray]]
    scalar_buffers: dict[str, list[float]]
    scalar_group_buffers: dict[str, dict[str, list[float]]]

    def __init__(self, output_dir: Path, batch_size: int) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.batch_count = 0
        self.total_saved = 0

        self.time_buffer = []
        self.field_buffers = {}
        self.scalar_buffers = {}
        self.scalar_group_buffers = {}

    def add_to_batch(self, t: float, nodes: list, **context) -> None:
        self.time_buffer.append(t)

        for node in nodes:
            if not node.save:
                continue

            field, reduced = node.compute(**context)
            if node.is_field:
                if isinstance(field, np.ndarray) and field.ndim == 2:
                    if node.name not in self.field_buffers:
                        self.field_buffers[node.name] = []
                    self.field_buffers[node.name].append(field)
            else:
                if node.name not in self.scalar_buffers:
                    self.scalar_buffers[node.name] = []
                self.scalar_buffers[node.name].append(float(reduced))

        scalar_groups = context.get("scalar_groups", {})
        for group_name, values in scalar_groups.items():
            if group_name not in self.scalar_group_buffers:
                self.scalar_group_buffers[group_name] = {}
            for label, value in values.items():
                if label not in self.scalar_group_buffers[group_name]:
                    self.scalar_group_buffers[group_name][label] = []
                self.scalar_group_buffers[group_name][label].append(float(value))

        self.batch_count += 1

        if self.batch_count >= self.batch_size:
            self.dump_batch()

    @abstractmethod
    def dump_batch(self) -> None:
        pass

    @abstractmethod
    def finalize(self) -> None:
        pass

    def _clear_buffers(self) -> None:
        self.time_buffer.clear()
        self.field_buffers.clear()
        self.scalar_buffers.clear()
        self.scalar_group_buffers.clear()
        self.batch_count = 0
