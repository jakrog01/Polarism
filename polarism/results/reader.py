"""Read saved HDF5 and NumPy simulation results."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np


_OLD_NAMES = {"|ψ|²": "density", "N(t)": "norm", "Pump": "pump"}


def _name(name: str) -> str:
    return _OLD_NAMES.get(name, name)


class _LazyDatasets(Mapping):
    """Expose HDF5 datasets by name; slice them to read only part of a file."""

    def __init__(self, group):
        self._group = group
        self._names = {_name(name): name for name in group} if group is not None else {}

    def __getitem__(self, name):
        return self._group[self._names[name]]

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


class Results:
    """Saved arrays and metadata, with an optional open HDF5 handle."""

    def __init__(self, time, fields, scalars, scalar_groups, static, config, x, y, attrs,
                 handle=None):
        self.time = time
        self.fields = fields
        self.scalars = scalars
        self.scalar_groups = scalar_groups
        self.static = static
        self.config = config
        self.x = x
        self.y = y
        self.attrs = attrs
        self._handle = handle

    def close(self):
        """Close an open HDF5 file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _load_hdf5(path: Path) -> Results:
    handle = h5py.File(path, "r")
    try:
        attrs = dict(handle.attrs)
        config = json.loads(handle["config"].attrs["json"]) if "config" in handle else {}
        grid = handle.get("grid")
        x = grid["x"][:] if grid is not None and "x" in grid else None
        y = grid["y"][:] if grid is not None and "y" in grid else None
        groups = handle.get("scalar_groups")
        scalar_groups = {name: _LazyDatasets(group) for name, group in groups.items()} if groups else {}
        return Results(
            handle["time"][:], _LazyDatasets(handle.get("fields")),
            _LazyDatasets(handle.get("scalars")), scalar_groups,
            _LazyDatasets(handle.get("static")), config, x, y, attrs, handle,
        )
    except BaseException:
        handle.close()
        raise


def _load_npy(path: Path) -> Results:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    batches = sorted(path.glob("batch_*.npz"))
    if not batches:
        raise FileNotFoundError(f"No NumPy result batches in {path}")
    arrays = {}
    for batch in batches:
        with np.load(batch) as data:
            for name in data.files:
                arrays.setdefault(name, []).append(data[name])
    combined = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    fields = {_name(key[6:]): value for key, value in combined.items() if key.startswith("field_")}
    scalars = {_name(key[7:]): value for key, value in combined.items() if key.startswith("scalar_")}
    scalar_groups = {}
    for key, value in combined.items():
        if key.startswith("group_"):
            group, label = key[6:].rsplit("_", 1)
            scalar_groups.setdefault(group, {})[label] = value
    static = {_name(file.stem[7:]): np.load(file) for file in path.glob("static_*.npy")}
    x_path, y_path = path / "grid_x.npy", path / "grid_y.npy"
    return Results(combined["time"], fields, scalars, scalar_groups, static,
                   metadata.get("config", {}), np.load(x_path) if x_path.exists() else None,
                   np.load(y_path) if y_path.exists() else None,
                   {key: value for key, value in metadata.items() if key != "config"})


def load_results(path: str | Path) -> Results:
    """Load one result run from its directory or HDF5 file."""
    path = Path(path)
    if path.is_file():
        if path.suffix not in (".h5", ".hdf5"):
            raise ValueError(f"Expected an HDF5 result file: {path}")
        return _load_hdf5(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    if (path / "results.h5").exists():
        return _load_hdf5(path / "results.h5")
    return _load_npy(path)
