"""Trim saved frames that are newer than a restored checkpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


def _keep_count(times, checkpoint_time: float) -> int:
    """Count the leading frames at or before the checkpoint."""
    return int(np.searchsorted(np.asarray(times), checkpoint_time, side="right"))


def _trim_hdf5(path: Path, checkpoint_time: float) -> None:
    """Resize every frame dataset while leaving static and grid data alone."""
    if not path.exists():
        return
    with h5py.File(path, "r+") as handle:
        version = handle.attrs.get("format_version")
        if version != 3:
            raise ValueError(f"results format v{version} is not supported by Polarism 0.7; supported: 3")
        if "time" not in handle:
            return
        keep = _keep_count(handle["time"][:], checkpoint_time)

        def trim(name, item):
            if not isinstance(item, h5py.Dataset):
                return
            if name != "time" and not name.startswith(("fields/", "scalars/", "scalar_groups/")):
                return
            if item.shape[0] < keep:
                raise ValueError(f"Saved dataset {name!r} is shorter than the checkpoint")
            if item.shape[0] != keep:
                item.resize((keep, *item.shape[1:]))

        handle.visititems(trim)
        handle.flush()


def _replace_npz(path: Path, arrays: dict, compressed: bool) -> None:
    """Atomically replace one partially retained NumPy batch."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.stem}.",
                                         suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            (np.savez_compressed if compressed else np.savez)(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _trim_npy(directory: Path, checkpoint_time: float, compressed: bool) -> None:
    """Remove later archives and shorten the archive crossing the checkpoint."""
    discard = False
    for path in sorted(directory.glob("batch_*.npz")):
        if discard:
            path.unlink()
            continue
        with np.load(path) as archive:
            times = archive["time"]
            keep = _keep_count(times, checkpoint_time)
            if keep == len(times):
                continue
            arrays = {name: archive[name][:keep] for name in archive.files} if keep else None
        if arrays is None:
            path.unlink()
        else:
            _replace_npz(path, arrays, compressed)
        discard = True


def _replace_json(path: Path, data: dict) -> None:
    """Atomically replace one partially retained JSON batch."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem}.", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _trim_json(directory: Path, checkpoint_time: float) -> None:
    """Trim JSON frame lists and retain static data in the first batch."""
    discard = False
    for path in sorted(directory.glob("batch_*.json")):
        if discard:
            path.unlink()
            continue
        data = json.loads(path.read_text())
        keep = _keep_count(data["time"], checkpoint_time)
        if keep == len(data["time"]):
            continue
        if keep == 0 and not data.get("static"):
            path.unlink()
        else:
            data["time"] = data["time"][:keep]
            for section in ("fields", "scalars"):
                for name, values in data[section].items():
                    data[section][name] = values[:keep]
            for values in data["scalar_groups"].values():
                for name, frames in values.items():
                    values[name] = frames[:keep]
            _replace_json(path, data)
        discard = True


def trim_saved_frames(directory: Path, checkpoint_time: float, names, *, compressed: bool) -> None:
    """Discard frames newer than the checkpoint for every enabled backend."""
    if "npy" in names or "json" in names:
        metadata_path = directory / "metadata.json"
        existing = (("npy" in names and any(directory.glob("batch_*.npz")))
                    or ("json" in names and any(directory.glob("batch_*.json"))))
        if existing:
            version = (json.loads(metadata_path.read_text()).get("format_version")
                       if metadata_path.exists() else None)
            if version != 3:
                raise ValueError(f"results format v{version} is not supported by Polarism 0.7; supported: 3")
    if "hdf5" in names:
        _trim_hdf5(directory / "results.h5", checkpoint_time)
    if "npy" in names:
        _trim_npy(directory, checkpoint_time, compressed)
    if "json" in names:
        _trim_json(directory, checkpoint_time)
