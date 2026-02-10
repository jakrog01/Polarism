from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from polarism.results.storage.base_storage import BaseStorage


class HDF5Storage(BaseStorage):
    h5_path: Path
    h5_file: h5py.File
    h5_datasets: dict[str, h5py.Dataset]

    def __init__(self, output_dir: Path, batch_size: int):
        super().__init__(output_dir, batch_size)
        self.h5_path = self.output_dir / "results.h5"
        self.h5_file = h5py.File(self.h5_path, "w")
        self.h5_datasets = {}

    def dump_batch(self) -> None:
        if not self.time_buffer:
            return

        batch_len = len(self.time_buffer)

        self._write_dataset("time", np.array(self.time_buffer), is_scalar=True)

        for field_name, field_list in self.field_buffers.items():
            field_array = np.array(field_list)
            self._write_dataset(f"fields/{field_name}", field_array, is_scalar=False)

        for scalar_name, scalar_list in self.scalar_buffers.items():
            scalar_array = np.array(scalar_list)
            self._write_dataset(f"scalars/{scalar_name}", scalar_array, is_scalar=True)

        for group_name, group_dict in self.scalar_group_buffers.items():
            for label, value_list in group_dict.items():
                value_array = np.array(value_list)
                self._write_dataset(
                    f"scalar_groups/{group_name}/{label}", value_array, is_scalar=True
                )

        self.total_saved += batch_len
        self._clear_buffers()

        self.h5_file.flush()

    def _write_dataset(self, name: str, data: np.ndarray, is_scalar: bool) -> None:
        if name not in self.h5_datasets:
            if is_scalar:
                self.h5_datasets[name] = self.h5_file.create_dataset(
                    name,
                    data=data,
                    maxshape=(None,),
                    compression="gzip",
                    compression_opts=4,
                    chunks=(min(1000, self.batch_size),),
                )
            else:
                shape = data.shape
                self.h5_datasets[name] = self.h5_file.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *shape[1:]),
                    compression="gzip",
                    compression_opts=4,
                    chunks=(1, *shape[1:]),
                )
        else:
            ds = self.h5_datasets[name]
            old_size = ds.shape[0]
            new_size = old_size + data.shape[0]

            ds.resize((new_size, *ds.shape[1:]))
            ds[old_size:new_size] = data

    def finalize(self) -> None:
        if self.batch_count > 0:
            self.dump_batch()

        if self.h5_file is not None:
            self.h5_file.close()
