"""HDF5 result storage."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from polarism.results.storage.base_storage import BaseStorage
from polarism.results.storage.storage_registry import register_storage


@register_storage("hdf5")
class HDF5Storage(BaseStorage):
    """Store result batches in HDF5 files."""
    h5_path: Path
    h5_file: h5py.File
    h5_datasets: dict[str, h5py.Dataset]

    def __init__(
        self,
        output_dir: Path,
        batch_size: int,
        config: Any | None = None,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        """Open the HDF5 output file."""
        super().__init__(output_dir, batch_size)
        self.ndim = config.grid.ndim if config is not None else (2 if y is not None else 1)
        self.components = 2 if config is not None and config.physics.spin == "spinor" else 1
        self.reservoir_components = (config.reservoir.components or self.components) if config is not None else 1
        self.h5_path = self.output_dir / "results.h5"
        self.h5_file = h5py.File(self.h5_path, "a" if getattr(config, "_storage_resume", False) else "w")
        self.h5_datasets = {}
        from polarism import __version__
        if getattr(config, "_storage_resume", False) and "format_version" in self.h5_file.attrs and self.h5_file.attrs["format_version"] != 3:
            found = self.h5_file.attrs["format_version"]
            raise ValueError(f"results format v{found} is not supported by this Polarism version; supported: 3")
        self.h5_file.attrs["format_version"] = 3
        self.h5_file.attrs["ndim"] = self.ndim
        self.h5_file.attrs["grid_type"] = config.grid.grid_type if config is not None else ""
        self.h5_file.attrs["components"] = (["plus", "minus"] if self.components == 2 else ["scalar"])
        self.h5_file.attrs["units"] = json.dumps({"x": "µm", "y": "µm", "time": "ps"})
        self.members = getattr(config, "_storage_members", None)
        if self.members is not None:
            self.h5_file.attrs["members"] = self.members
            if "member_index" not in self.h5_file:
                dataset = self.h5_file.create_dataset("member_index", data=np.asarray(config._storage_indices))
                dataset.attrs["dims"] = ["member"]
                dataset.attrs["units"] = ""
        self.h5_file.attrs["polarism_version"] = __version__
        for axis, values in (("x", x), ("y", y)):
            if values is not None:
                if f"grid/{axis}" in self.h5_file:
                    continue
                dataset = self.h5_file.create_dataset(f"grid/{axis}", data=np.asarray(values))
                dataset.attrs["units"] = "µm"
                dataset.attrs["dims"] = [axis]
        self.compression = getattr(config.result, "compression", "none") if config is not None else "none"
        self.compression_level = getattr(config.result, "compression_level", 4) if config is not None else 4
        if config is not None:
            config_group = self.h5_file.require_group("config")
            config_group.attrs["json"] = json.dumps(
                asdict(config) if is_dataclass(config) else config,
                sort_keys=True,
            )
            member_seeds = getattr(config, "_storage_member_seeds", None)
            if member_seeds is not None and "member_seeds" not in config_group:
                config_group.create_dataset("member_seeds", data=np.asarray(member_seeds, dtype=np.int64))

    def dump_batch(self) -> None:
        """Write the current batch to HDF5."""
        if not self.time_buffer:
            return

        batch_len = len(self.time_buffer)

        self._write_dataset("time", np.array(self.time_buffer), is_scalar=True)

        for field_name, field_list in self.field_buffers.items():
            field_array = np.array(field_list)
            component_axis = 2 if self.members is not None else 1
            has_components = (self.h5_file.attrs["components"].shape[0] == 2
                              and field_name != "stokes"
                              and field_array.ndim == self.ndim + (3 if self.members is not None else 2))
            if not has_components and field_name != "stokes":
                field_array = np.expand_dims(field_array, axis=component_axis)
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
        """Write one dataset to the HDF5 file."""
        compression = None if self.compression == "none" else self.compression
        options = self.compression_level if compression == "gzip" else None
        if name not in self.h5_datasets:
            if name in self.h5_file:
                self.h5_datasets[name] = self.h5_file[name]
                self._write_dataset(name, data, is_scalar)
                return
            if is_scalar:
                self.h5_datasets[name] = self.h5_file.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *data.shape[1:]),
                    compression=compression,
                    compression_opts=options,
                    shuffle=compression is not None,
                    chunks=(min(1000, self.batch_size), 1) if self.members is not None and data.ndim == 2 else (min(1000, self.batch_size),),
                )
            else:
                shape = data.shape
                self.h5_datasets[name] = self.h5_file.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *shape[1:]),
                    compression=compression,
                    compression_opts=options,
                    shuffle=compression is not None,
                    chunks=(min(16, self.batch_size), 1, *shape[2:]) if self.members is not None else (1, *shape[1:]),
                )
            leaf = name.split("/")[-1]
            if name == "time":
                metadata = {"label": "time", "units": "ps", "dims": ["time"]}
            else:
                metadata = self.dataset_metadata(leaf, field=not is_scalar)
                if not is_scalar and leaf != "stokes" and "component" not in metadata["dims"]:
                    metadata["dims"].insert(2 if self.members is not None else 1, "component")
            self.h5_datasets[name].attrs.update(metadata)
        else:
            ds = self.h5_datasets[name]
            old_size = ds.shape[0]
            new_size = old_size + data.shape[0]

            ds.resize((new_size, *ds.shape[1:]))
            ds[old_size:new_size] = data

    def write_static(self, name: str, field: np.ndarray) -> None:
        """Store a field once outside the frame axis."""
        compression = None if self.compression == "none" else self.compression
        options = self.compression_level if compression == "gzip" else None
        if f"static/{name}" in self.h5_file:
            return
        dataset = self.h5_file.create_dataset(
            f"static/{name}", data=field, compression=compression,
            compression_opts=options, shuffle=compression is not None,
        )
        metadata = self.dataset_metadata(name, static=True, field=True)
        if self.members is not None and field.ndim == self.ndim:
            metadata["dims"].remove("member")
        dataset.attrs.update(metadata)

    def finalize(self) -> None:
        """Flush remaining data and close the HDF5 file."""
        if self.batch_count > 0:
            self.dump_batch()

        if self.h5_file is not None:
            self.h5_file.close()

    def flush(self) -> None:
        """Persist pending frames and flush the open HDF5 handle."""
        super().flush()
        self.h5_file.flush()

    def set_stop_times(self, stop_times) -> None:
        """Store stop times in root metadata, aligned with member_index."""
        self.h5_file.attrs["stop_time"] = np.asarray(stop_times, dtype=float)
        self.h5_file.flush()

    def set_step_statistics(self, statistics) -> None:
        """Write adaptive trial statistics as root attributes."""
        for name, value in statistics.items():
            self.h5_file.attrs[f"adaptive_{name}"] = value
        self.h5_file.flush()

    def write_accumulated(self, results, metadata) -> None:
        """Write final accumulated datasets with sampling metadata."""
        root = self.h5_file.require_group("accumulated")
        for name, result in results.items():
            group = root.require_group(name)
            if "value" in group:
                del group["value"]
            dataset = group.create_dataset("value", data=result.value)
            dataset.attrs["dims"] = result.dims
            dataset.attrs["units"] = result.units
            dataset.attrs["label"] = result.label
            dataset.attrs["wigner_corrected"] = result.wigner_corrected
            dataset.attrs["linear"] = result.linear
            group.attrs.update(metadata[name])
        self.h5_file.flush()
