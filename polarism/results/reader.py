"""Read saved HDF5 and NumPy simulation results."""
from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np


class _LazyDatasets(Mapping):
    """Expose HDF5 datasets by name; slice them to read only part of a file."""

    def __init__(self, group, *, field=False):
        self._group = group
        self._field = field
        self._names = {name: name for name in group} if group is not None else {}

    def __getitem__(self, name):
        dataset = self._group[self._names[name]]
        return _ScalarComponentDataset(dataset) if self._field else dataset

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


class _ScalarComponentDataset:
    """Expose a canonical scalar field in its public shape without copying."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.attrs = dataset.attrs
        self.dtype = dataset.dtype
        self._component_axis = 2 if "member" in dataset.attrs["dims"] else 1
        self.shape = tuple(size for axis, size in enumerate(dataset.shape)
                           if axis != self._component_axis)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = list(key)
        if Ellipsis in key:
            index = key.index(Ellipsis)
            key[index:index + 1] = [slice(None)] * (self.ndim - len(key) + 1)
        key += [slice(None)] * (self._component_axis - len(key))
        key.insert(self._component_axis, 0)
        return self.dataset[tuple(key)]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)


class _MemberDataset:
    """Read one member's frames from its batch dataset."""

    def __init__(self, dataset, position, static=False, spatial_ndim=2):
        self.dataset = dataset
        self.position = position
        self.static = static
        self._static_member = static and ("member" in getattr(dataset, "attrs", {}).get("dims", ())
                                          or not hasattr(dataset, "attrs") and dataset.ndim == spatial_ndim + 1)
        self.shape = dataset.shape[1:] if self._static_member else (
            dataset.shape if static else (dataset.shape[0], *dataset.shape[2:])
        )
        self.dtype = dataset.dtype
        self.attrs = getattr(dataset, "attrs", {})

    def __getitem__(self, key):
        if self.static:
            return self.dataset[self.position][key] if self._static_member else self.dataset[key]
        return self.dataset[key, self.position]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)


class _SweepDataset:
    """Concatenate member slices only when requested by the caller."""

    def __init__(self, datasets, locations):
        self.datasets = datasets
        self.locations = locations
        first = datasets[0]
        if any(dataset.shape[0] != first.shape[0] or dataset.shape[2:] != first.shape[2:]
               for dataset in datasets[1:]):
            raise ValueError("Sweep batches have incompatible frame or grid shapes")
        self.shape = (first.shape[0], len(locations), *first.shape[2:])
        self.dtype = first.dtype
        self.attrs = getattr(first, "attrs", {})

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        frame = key[0]
        member_key = key[1] if len(key) > 1 else slice(None)
        spatial = key[2:]
        positions = np.arange(len(self.locations))[member_key]
        if np.isscalar(positions):
            batch, local = self.locations[int(positions)]
            return self.datasets[batch][(frame, local, *spatial)]
        values = [self.datasets[batch][(frame, local, *spatial)]
                  for batch, local in (self.locations[int(i)] for i in positions)]
        axis = 0 if isinstance(frame, (int, np.integer)) else 1
        return np.stack(values, axis=axis)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)


class _SweepMapping(Mapping):
    """Map names to lazy datasets spanning all sweep batches."""

    def __init__(self, results, locations, attribute):
        self.results = results
        self.locations = locations
        self.attribute = attribute

    def __getitem__(self, name):
        return _SweepDataset([getattr(result, self.attribute)[name] for result in self.results],
                             self.locations)

    def __iter__(self):
        return iter(getattr(self.results[0], self.attribute))

    def __len__(self):
        return len(getattr(self.results[0], self.attribute))


class SweepResults:
    """Read a sweep manifest and provide lazy member and combined result views."""

    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "sweep.json").read_text())
        self.parameters = self.manifest["parameters"]
        self.parameter_table = {key: np.asarray([row[key] for row in self.parameters])
                                for key in self.manifest["parameter_keys"]}
        self._results = [load_results(path / f"batch_{i:03d}")
                         for i in range(len(self.manifest["batches"]))]
        self._locations = [(entry["batch"], entry["position"])
                           for entry in self.manifest["members"]]
        self.fields = _SweepMapping(self._results, self._locations, "fields")
        self.scalars = _SweepMapping(self._results, self._locations, "scalars")

    def __len__(self):
        return len(self.parameters)

    def __iter__(self):
        return (self.member(i) for i in range(len(self)))

    def member(self, index: int) -> Results:
        """Return a Results-compatible view of one global member index."""
        batch, position = self._locations[index]
        result = self._results[batch]
        config = deepcopy(result.config)
        for path, value in self.parameters[index].items():
            if path.startswith("lasers["):
                continue
            parts = path.split(".")
            target = config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        select = lambda mapping, static=False: {
            name: _MemberDataset(value, position, static=static,
                                 spatial_ndim=int(result.attrs.get("ndim", 2)))
            for name, value in mapping.items()
        }
        return Results(result.time, select(result.fields), select(result.scalars),
                       {name: select(group) for name, group in result.scalar_groups.items()},
                       select(result.static, static=True), config, result.x, result.y,
                       {**result.attrs, "member_index": index,
                        "parameters": self.parameters[index]})

    def close(self):
        """Close every underlying batch result file."""
        for result in self._results:
            result.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _load_hdf5(path: Path) -> Results:
    handle = h5py.File(path, "r")
    try:
        attrs = dict(handle.attrs)
        version = attrs.get("format_version")
        if version != 3:
            raise ValueError(f"results format v{version} is not supported by Polarism 0.7; supported: 3")
        config = json.loads(handle["config"].attrs["json"]) if "config" in handle else {}
        grid = handle.get("grid")
        x = grid["x"][:] if grid is not None and "x" in grid else None
        y = grid["y"][:] if grid is not None and "y" in grid else None
        groups = handle.get("scalar_groups")
        scalar_groups = {name: _LazyDatasets(group) for name, group in groups.items()} if groups else {}
        return Results(
            handle["time"][:], _LazyDatasets(handle.get("fields"), field=True),
            _LazyDatasets(handle.get("scalars")), scalar_groups,
            _LazyDatasets(handle.get("static")), config, x, y, attrs, handle,
        )
    except BaseException:
        handle.close()
        raise


def _load_npy(path: Path) -> Results:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    version = metadata.get("format_version")
    if version != 3:
        raise ValueError(f"results format v{version} is not supported by Polarism 0.7; supported: 3")
    batches = sorted(path.glob("batch_*.npz"))
    if not batches:
        raise FileNotFoundError(f"No NumPy result batches in {path}")
    arrays = {}
    for batch in batches:
        with np.load(batch) as data:
            for name in data.files:
                arrays.setdefault(name, []).append(data[name])
    combined = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    fields = {key[6:]: value for key, value in combined.items() if key.startswith("field_")}
    scalars = {key[7:]: value for key, value in combined.items() if key.startswith("scalar_")}
    scalar_groups = {}
    for key, value in combined.items():
        if key.startswith("group_"):
            group, label = key[6:].rsplit("_", 1)
            scalar_groups.setdefault(group, {})[label] = value
    static = {file.stem[7:]: np.load(file) for file in path.glob("static_*.npy")}
    x_path, y_path = path / "grid_x.npy", path / "grid_y.npy"
    return Results(combined["time"], fields, scalars, scalar_groups, static,
                   metadata.get("config", {}), np.load(x_path) if x_path.exists() else None,
                   np.load(y_path) if y_path.exists() else None,
                   {key: value for key, value in metadata.items() if key != "config"})


def _load_json(path: Path) -> Results:
    batches = sorted(path.glob("batch_*.json"))
    if not batches:
        raise FileNotFoundError(f"No JSON result batches in {path}")
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    version = metadata.get("format_version")
    if version != 3:
        raise ValueError(f"results format v{version} is not supported by Polarism 0.7; supported: 3")
    data = [json.loads(file.read_text()) for file in batches]

    def decode(value):
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return np.stack([decode(item) for item in value])
        if isinstance(value, dict) and "real" in value and "imag" in value:
            return np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
        return np.asarray(value)

    def combine(section):
        return {name: np.concatenate([decode(batch[section][name]) for batch in data
                                      if batch["time"]])
                for name in data[0][section]}

    fields = combine("fields")
    scalars = combine("scalars")
    groups = {name: {label: np.concatenate([decode(batch["scalar_groups"][name][label])
                                        for batch in data if batch["time"]]) for label in labels}
              for name, labels in data[0]["scalar_groups"].items()}
    static = {name: decode(value) for name, value in data[0].get("static", {}).items()}
    x_path, y_path = path / "grid_x.json", path / "grid_y.json"
    return Results(np.concatenate([np.asarray(batch["time"]) for batch in data]),
                   fields, scalars, groups, static, metadata.get("config", {}),
                   np.asarray(json.loads(x_path.read_text())) if x_path.exists() else None,
                   np.asarray(json.loads(y_path.read_text())) if y_path.exists() else None,
                   {key: value for key, value in metadata.items() if key != "config"})


def load_results(path: str | Path) -> Results | SweepResults:
    """Load one result run from its directory or HDF5 file."""
    path = Path(path)
    if path.is_file():
        if path.suffix not in (".h5", ".hdf5"):
            raise ValueError(f"Expected an HDF5 result file: {path}")
        return _load_hdf5(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    if (path / "sweep.json").exists():
        return SweepResults(path)
    if (path / "results.h5").exists():
        return _load_hdf5(path / "results.h5")
    if not list(path.glob("batch_*.npz")):
        return _load_json(path)
    return _load_npy(path)
