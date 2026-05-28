"""Run-directory manifest and atomic file I/O for the characteristic map pipeline.

Per-point results are written to individual JSON files under
``run_dir/points/`` rather than a shared manifest, enabling concurrent
Slurm array tasks to write without coordination.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Any

MANIFEST_FILE = "manifest.json"
POINT_INDEX_FILE = "point_index.json"


def atomic_write_json(path: str, data: Any) -> None:
    """Write *data* as JSON to *path* atomically via a sibling temp file."""
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, path)


def _config_hash(config_path: str) -> str:
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:8]


def create_run_dir(base_dir: str, config_path: str, prefix: str = "cc") -> str:
    """Create a unique timestamped run directory under *base_dir*."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chash = _config_hash(config_path)
    run_dir = os.path.join(os.path.abspath(base_dir), f"{prefix}_{ts}_{chash}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def init_manifest(run_dir: str, config_path: str, n_points: int) -> None:
    """Write the initial manifest for a new characteristic map run."""
    data: dict[str, Any] = {
        "run_dir": run_dir,
        "config_path": config_path,
        "n_points": n_points,
        "sweep_complete": False,
        "finalize_complete": False,
    }
    atomic_write_json(os.path.join(run_dir, MANIFEST_FILE), data)


def set_manifest_field(run_dir: str, key: str, value: Any) -> None:
    """Atomically update one top-level key in the manifest."""
    path = os.path.join(run_dir, MANIFEST_FILE)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data[key] = value
    atomic_write_json(path, data)


def load_point_index(run_dir: str) -> list[dict[str, Any]]:
    """Return the ordered list of grid points from *point_index.json*."""
    path = os.path.join(run_dir, POINT_INDEX_FILE)
    with open(path) as f:
        return json.load(f)


def point_result_path(run_dir: str, point_index: int) -> str:
    """Canonical path for a per-point scalar result JSON."""
    return os.path.join(run_dir, "points", f"point_{point_index:06d}.json")


def load_point_result(run_dir: str, point_index: int) -> dict[str, Any]:
    """Load a per-point scalar result JSON."""
    with open(point_result_path(run_dir, point_index)) as f:
        return json.load(f)
