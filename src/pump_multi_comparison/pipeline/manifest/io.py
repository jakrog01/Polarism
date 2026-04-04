"""Run-scoped manifest and atomic file I/O.

Every write goes through :func:`atomic_write_json`, which writes a temp
file, fsyncs it, and then swaps it into place.

The manifest file is the main record for a run. Individual scenario jobs
write sidecar files instead of updating the manifest at the same time.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from typing import Any


# Filename constants shared across all pipeline stages.
MANIFEST_FILE = "manifest.json"
SCENARIO_INDEX_FILE = "scenario_index.json"


# ── Atomic I/O ────────────────────────────────────────────────────────────────

def atomic_write_json(path: str, data: Any) -> None:
    """Write *data* as JSON to *path* atomically.

    Strategy: write to a sibling ``.tmp`` file, fsync, then ``os.replace``
    into the final path.  ``os.replace`` is atomic on POSIX within the same
    filesystem, which holds for all paths within a single run directory.
    """
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


# ── Run directory ─────────────────────────────────────────────────────────────

def config_hash(config_path: str) -> str:
    """First 8 hex chars of the SHA-256 of the config file content."""
    h = hashlib.sha256()
    with open(config_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:8]


def create_run_dir(base_dir: str, config_path: str) -> str:
    """Create a unique, timestamped run directory under *base_dir*.

    The directory name encodes a timestamp and a content hash of
    *config_path* so two runs with different configs are always distinct.

    Returns
    -------
    str
        Absolute path to the newly created directory.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chash = config_hash(config_path)
    run_dir = os.path.join(os.path.abspath(base_dir), f"{ts}_{chash}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


# ── Manifest ──────────────────────────────────────────────────────────────────

def init_manifest(
    run_dir: str,
    config_path: str,
    scenarios: list[str],
) -> None:
    """Create the initial manifest for a new run.

    Called once by ``submit.sh`` before any jobs are submitted.
    """
    data: dict[str, Any] = {
        "run_dir": run_dir,
        "config_path": config_path,
        "scenarios": scenarios,
        "threshold_complete": False,
        "postprocess_complete": False,
        "finalize_complete": False,
    }
    atomic_write_json(os.path.join(run_dir, MANIFEST_FILE), data)


def load_manifest(run_dir: str) -> dict[str, Any]:
    """Load and return the manifest dict."""
    with open(os.path.join(run_dir, MANIFEST_FILE)) as f:
        return json.load(f)


def set_manifest_field(run_dir: str, key: str, value: Any) -> None:
    """Atomically update one top-level key in the manifest.

    Safe for single-writer use within one job.  For concurrent multi-job
    writes use per-scenario sidecar files instead.
    """
    path = os.path.join(run_dir, MANIFEST_FILE)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    data[key] = value
    atomic_write_json(path, data)


def load_scenario_index(run_dir: str) -> list[str]:
    """Return the ordered list of scenario names from *scenario_index.json*."""
    path = os.path.join(run_dir, SCENARIO_INDEX_FILE)
    with open(path) as f:
        return json.load(f)


def resolve_scenario_name(run_dir: str, scenario_index: int) -> str:
    """Map an array task index to a scenario name.

    Parameters
    ----------
    run_dir : str
        Path to the run directory (contains ``scenario_index.json``).
    scenario_index : int
        Zero-based index, typically ``$SLURM_ARRAY_TASK_ID``.

    Returns
    -------
    str
        Scenario name.

    Raises
    ------
    IndexError
        If *scenario_index* is out of range.
    FileNotFoundError
        If ``scenario_index.json`` is absent.
    """
    names = load_scenario_index(run_dir)
    if scenario_index >= len(names):
        raise IndexError(
            f"scenario_index={scenario_index} out of range (have {len(names)} scenarios)"
        )
    return names[scenario_index]


# ── Scenario sidecar files ────────────────────────────────────────────────────

def scenario_meta_path(run_dir: str, scenario_name: str) -> str:
    """Return the canonical path for a scenario's metadata sidecar file."""
    return os.path.join(run_dir, f"{scenario_name}_meta.json")


def load_scenario_meta(run_dir: str, scenario_name: str) -> dict[str, Any]:
    """Load scenario metadata written by the GPU simulation stage."""
    with open(scenario_meta_path(run_dir, scenario_name)) as f:
        return json.load(f)
