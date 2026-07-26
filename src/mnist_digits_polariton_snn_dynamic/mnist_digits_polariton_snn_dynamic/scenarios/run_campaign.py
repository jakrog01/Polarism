"""Run one campaign scenario."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json
from mnist_digits_polariton_snn_dynamic.pipeline import run_pipeline
from mnist_digits_polariton_snn_dynamic.simulation.calibration import (
    final_power_max,
    is_scenario_skipped,
)


def main() -> None:
    """Run one scenario selected from a campaign manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaign-output-dir", required=True)
    parser.add_argument("--use-calibrated-power", action="store_true")
    args = parser.parse_args()
    run_scenario(
        args.manifest,
        args.scenario_id,
        args.campaign_output_dir,
        use_calibrated_power=args.use_calibrated_power,
    )


def run_scenario(
    manifest_path: str,
    scenario_id: str,
    campaign_output_dir: str,
    use_calibrated_power: bool = False,
) -> None:
    """Run one scenario from a manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_data = _load_yaml(manifest)
    scenario = _find_scenario(manifest_data, scenario_id)
    config = (manifest.parent / str(scenario["config"])).resolve()
    output_dir = Path(campaign_output_dir).expanduser().resolve() / scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)
    power_max = None
    if use_calibrated_power:
        scenario_calibration = output_dir / "calibration.json"
        skipped, reason = is_scenario_skipped(str(scenario_calibration))
        if skipped:
            _write_skip_marker(
                output_dir / "skipped.json",
                scenario_id,
                str(scenario_calibration),
                reason,
                stage="run",
            )
            print(
                f"SKIP: scenario {scenario_id!r} run stage skipped: {reason}",
                file=sys.stderr,
                flush=True,
            )
            now = datetime.now(timezone.utc)
            _write_scenario_meta(
                output_dir / "scenario_meta.json",
                scenario_id,
                config,
                manifest,
                now,
                now,
                error=None,
                skipped=True,
                skip_reason=reason,
            )
            return
        power_max = final_power_max(str(scenario_calibration))
        _write_calibrated_power(
            output_dir / "calibrated_power.json",
            scenario_calibration,
            power_max,
        )
    start = datetime.now(timezone.utc)
    error: str | None = None
    try:
        run_pipeline(str(config), output_dir=str(output_dir), power_max_override=power_max)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        stop = datetime.now(timezone.utc)
        _write_scenario_meta(
            output_dir / "scenario_meta.json",
            scenario_id,
            config,
            manifest,
            start,
            stop,
            error,
        )


def _write_skip_marker(
    path: Path,
    scenario_id: str,
    calibration_path: str,
    reason: str | None,
    stage: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        str(path),
        {
            "scenario_id": scenario_id,
            "stage": stage,
            "calibration_path": calibration_path,
            "reason": reason,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _find_scenario(manifest_data: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = manifest_data.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("manifest.scenarios must be a list")
    for scenario in scenarios:
        if isinstance(scenario, dict) and scenario.get("id") == scenario_id:
            if "config" not in scenario:
                raise ValueError(f"Scenario {scenario_id!r} is missing config")
            return scenario
    raise ValueError(f"Scenario not found in manifest: {scenario_id}")


def _write_calibrated_power(
    path: Path,
    scenario_calibration_path: Path,
    power_max: float,
) -> None:
    scenario = _load_json(scenario_calibration_path)
    atomic_write_json(
        str(path),
        {
            "scenario_calibration_path": str(scenario_calibration_path),
            "threshold_power_full_lattice": float(
                scenario["threshold_power_full_lattice"]
            ),
            "threshold_power_single_spot": scenario["threshold_power_single_spot"],
            "power_max_candidate": float(scenario["power_max_candidate"]),
            "final_power_max": float(power_max),
            "power_max_was_capped": bool(scenario["power_max_was_capped"]),
        },
    )


def _write_scenario_meta(
    path: Path,
    scenario_id: str,
    config: Path,
    manifest: Path,
    start: datetime,
    stop: datetime,
    error: str | None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> None:
    meta = {
        "scenario_id": scenario_id,
        "config_path": str(config),
        "config_sha256": _sha256_file(config),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "started_at_utc": start.isoformat(),
        "stopped_at_utc": stop.isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "error": error,
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain a mapping: {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
