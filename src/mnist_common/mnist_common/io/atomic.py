"""Atomic file-write helpers for JSON and NPZ."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np


class _NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def atomic_write_json(path: str, data: Any) -> None:
    """Write data as JSON atomically (temp file + os.replace)."""
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, cls=_NpEncoder)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def atomic_write_npz(path: str, **arrays: np.ndarray) -> None:
    """Write arrays to a compressed .npz file atomically."""
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
