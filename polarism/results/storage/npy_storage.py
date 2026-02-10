from __future__ import annotations

from pathlib import Path

import numpy as np

from polarism.results.storage.base_storage import BaseStorage


class NPYStorage(BaseStorage):
    npz_path: Path
    all_time: list[float]
    all_fields: dict[str, list[np.ndarray]]
    all_scalars: dict[str, list[float]]
    all_scalar_groups: dict[str, list[float]]

    def __init__(self, output_dir: Path, batch_size: int):
        super().__init__(output_dir, batch_size)
        self.npz_path = self.output_dir / "results.npz"

        self.all_time = []
        self.all_fields = {}
        self.all_scalars = {}
        self.all_scalar_groups = {}

    def dump_batch(self) -> None:
        if not self.time_buffer:
            return

        self.all_time.extend(self.time_buffer)

        for field_name, field_list in self.field_buffers.items():
            if field_name not in self.all_fields:
                self.all_fields[field_name] = []
            self.all_fields[field_name].extend(field_list)

        for scalar_name, scalar_list in self.scalar_buffers.items():
            if scalar_name not in self.all_scalars:
                self.all_scalars[scalar_name] = []
            self.all_scalars[scalar_name].extend(scalar_list)

        for group_name, group_dict in self.scalar_group_buffers.items():
            for label, value_list in group_dict.items():
                key = f"{group_name}_{label}"
                if key not in self.all_scalar_groups:
                    self.all_scalar_groups[key] = []
                self.all_scalar_groups[key].extend(value_list)

        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def finalize(self) -> None:
        if self.batch_count > 0:
            self.dump_batch()

        arrays = {}

        arrays["time"] = np.array(self.all_time)

        for field_name, field_list in self.all_fields.items():
            arrays[f"field_{field_name}"] = np.stack(field_list, axis=0)

        for scalar_name, scalar_list in self.all_scalars.items():
            arrays[f"scalar_{scalar_name}"] = np.array(scalar_list)

        for key, value_list in self.all_scalar_groups.items():
            arrays[f"group_{key}"] = np.array(value_list)

        np.savez_compressed(self.npz_path, **arrays)
