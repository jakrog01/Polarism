"""Shared metadata helpers for scenario stages."""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def find_scenario(manifest_data: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    """Return the requested manifest scenario."""
    scenarios = manifest_data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("manifest.scenarios must be a list")
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("id") == scenario_id:
            if "config" not in scenario:
                raise ValueError(f"Scenario {scenario_id!r} is missing config")
            return scenario
    raise ValueError(f"Scenario not found in manifest: {scenario_id}")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stage_meta(
    path: Path,
    scenario_id: str,
    config: Path,
    manifest: Path,
    stage: str,
    start: datetime,
    stop: datetime,
    error: str | None,
) -> None:
    """Write common stage metadata atomically."""
    atomic_write_json(
        str(path),
        {
            "scenario_id": scenario_id,
            "stage": stage,
            "config_path": str(config),
            "config_sha256": sha256_file(config),
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "started_at_utc": start.isoformat(),
            "stopped_at_utc": stop.isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "error": error,
        },
    )
