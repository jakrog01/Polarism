from __future__ import annotations

import json
from pathlib import Path

from polarism.results.storage.base_storage import BaseStorage


class JSONStorage(BaseStorage):
    json_path: Path
    all_data: dict[str, list | dict]

    def __init__(self, output_dir: Path, batch_size: int):
        super().__init__(output_dir, batch_size)
        self.json_path = self.output_dir / "results.json"

        self.all_data = {
            "time": [],
            "fields": {},
            "scalars": {},
            "scalar_groups": {},
        }

    def dump_batch(self) -> None:
        if not self.time_buffer:
            return

        time_data = self.all_data["time"]
        if isinstance(time_data, list):
            time_data.extend(self.time_buffer)

        fields_data = self.all_data["fields"]
        if isinstance(fields_data, dict):
            for field_name, field_list in self.field_buffers.items():
                if field_name not in fields_data:
                    fields_data[field_name] = []

                field_data = fields_data[field_name]
                if isinstance(field_data, list):
                    for field in field_list:
                        field_data.append(field.tolist())

        scalars_data = self.all_data["scalars"]
        if isinstance(scalars_data, dict):
            for scalar_name, scalar_list in self.scalar_buffers.items():
                if scalar_name not in scalars_data:
                    scalars_data[scalar_name] = []
                scalar_values = scalars_data[scalar_name]
                if isinstance(scalar_values, list):
                    scalar_values.extend(scalar_list)

        scalar_groups_data = self.all_data["scalar_groups"]
        if isinstance(scalar_groups_data, dict):
            for group_name, group_dict in self.scalar_group_buffers.items():
                if group_name not in scalar_groups_data:
                    scalar_groups_data[group_name] = {}
                group_data = scalar_groups_data[group_name]
                if isinstance(group_data, dict):
                    for label, value_list in group_dict.items():
                        if label not in group_data:
                            group_data[label] = []
                        label_data = group_data[label]
                        if isinstance(label_data, list):
                            label_data.extend(value_list)

        self.total_saved += len(self.time_buffer)
        self._clear_buffers()

    def finalize(self) -> None:
        if self.batch_count > 0:
            self.dump_batch()
        with open(self.json_path, "w") as f:
            json.dump(self.all_data, f, indent=2)
