"""Shared result storage helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseStorage(ABC):
    """Store result batches in a common format."""
    output_dir: Path
    batch_size: int
    batch_count: int
    total_saved: int

    time_buffer: list[float]
    field_buffers: dict[str, list[np.ndarray]]
    scalar_buffers: dict[str, list[float]]
    scalar_group_buffers: dict[str, dict[str, list[float]]]

    def __init__(self, output_dir: Path, batch_size: int) -> None:
        """Set up common storage buffers."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.batch_count = 0
        self.total_saved = 0

        self.time_buffer = []
        self.field_buffers = {}
        self.scalar_buffers = {}
        self.scalar_group_buffers = {}
        self.static_fields: dict[str, np.ndarray] = {}
        self.node_metadata: dict[str, dict[str, str]] = {}
        self.members: int | None = None

    def add_to_batch(self, t: float, nodes: list, **context) -> None:
        """Add one result step to the current batch."""
        cached = context.get("cached", {})
        self.time_buffer.append(t)

        for node in nodes:
            if not node.save:
                continue
            self.node_metadata[node.name] = {
                "label": node.label or node.name,
                **({"units": node.units} if node.units is not None else {}),
            }

            field, reduced = cached[node.name]
            if node.is_field:
                if node.static:
                    if node.name not in self.static_fields:
                        static = field
                        if self.members is not None and field.ndim == 3 and all(
                            np.array_equal(field[0], item) for item in field[1:]
                        ):
                            static = field[0]
                        self.static_fields[node.name] = static
                        self.write_static(node.name, static)
                    continue
                if node.name not in self.field_buffers:
                    self.field_buffers[node.name] = []
                self.field_buffers[node.name].append(field)
            else:
                if node.name not in self.scalar_buffers:
                    self.scalar_buffers[node.name] = []
                value = field if reduced is None else reduced
                if self.members is not None and np.shape(value) != (self.members,):
                    raise ValueError(f"batched scalar {node.name!r} must have one value per member")
                self.scalar_buffers[node.name].append(
                    np.asarray(value) if self.members is not None else float(value)
                )

        scalar_groups = context.get("scalar_groups", {})
        for group_name, values in scalar_groups.items():
            if group_name not in self.scalar_group_buffers:
                self.scalar_group_buffers[group_name] = {}
            for label, value in values.items():
                if self.members is not None and np.shape(value) != (self.members,):
                    raise ValueError(f"batched scalar group {group_name}/{label} must have one value per member")
                if label not in self.scalar_group_buffers[group_name]:
                    self.scalar_group_buffers[group_name][label] = []
                self.scalar_group_buffers[group_name][label].append(
                    np.asarray(value) if self.members is not None else float(value)
                )

        self.batch_count += 1

        if self.batch_count >= self.batch_size:
            self.dump_batch()

    @abstractmethod
    def dump_batch(self) -> None:
        """Write the current batch to disk."""
        pass

    @abstractmethod
    def write_static(self, name: str, field: np.ndarray) -> None:
        """Write a field shared by every frame."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Flush remaining data and close resources."""
        pass

    def flush(self) -> None:
        """Persist all buffered frames without closing the storage."""
        if self.batch_count > 0:
            self.dump_batch()

    def _clear_buffers(self) -> None:
        """Clear all in-memory buffers."""
        self.time_buffer.clear()
        self.field_buffers.clear()
        self.scalar_buffers.clear()
        self.scalar_group_buffers.clear()
        self.batch_count = 0
