"""Run threshold calibration for one campaign scenario."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mnist_digits_polariton_snn_dynamic.simulation.calibration import calibrate_threshold


def main() -> None:
    """Run one scenario calibration selected from a campaign manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaign-output-dir", required=True)
    parser.add_argument("--n-points", type=int, default=17)
    parser.add_argument("--condensation-ratio", type=float, default=10.0)
    parser.add_argument("--full-lattice-power-margin", type=float, default=1.05)
    parser.add_argument("--single-spot-safety-fraction", type=float, default=0.95)
    parser.add_argument("--realistic-safety-fraction", type=float, default=0.95)
    parser.add_argument("--guard-n-points", type=int, default=10)
    args = parser.parse_args()
    run_calibration(
        args.manifest,
        args.scenario_id,
        args.campaign_output_dir,
        n_points=args.n_points,
        condensation_ratio=args.condensation_ratio,
        full_lattice_power_margin=args.full_lattice_power_margin,
        single_spot_safety_fraction=args.single_spot_safety_fraction,
        guard_n_points=args.guard_n_points,
        realistic_safety_fraction=args.realistic_safety_fraction,
    )


def run_calibration(
    manifest_path: str,
    scenario_id: str,
    campaign_output_dir: str,
    *,
    n_points: int,
    condensation_ratio: float,
    full_lattice_power_margin: float,
    single_spot_safety_fraction: float,
    guard_n_points: int,
    realistic_safety_fraction: float = 0.95,
) -> None:
    """Run threshold calibration for one scenario from a manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_data = _load_yaml(manifest)
    scenario = _find_scenario(manifest_data, scenario_id)
    config = (manifest.parent / str(scenario["config"])).resolve()
    output_dir = Path(campaign_output_dir).expanduser().resolve() / scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    error: str | None = None
    try:
        calibrate_threshold(
            str(config),
            scenario_id,
            str(output_dir),
            n_points=n_points,
            condensation_ratio=condensation_ratio,
            full_lattice_power_margin=full_lattice_power_margin,
            single_spot_safety_fraction=single_spot_safety_fraction,
            guard_n_points=guard_n_points,
            realistic_safety_fraction=realistic_safety_fraction,
        )
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        stop = datetime.now(timezone.utc)
        _write_stage_meta(
            output_dir / "calibration_meta.json",
            scenario_id,
            config,
            manifest,
            "calibrate",
            start,
            stop,
            error,
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


def _write_stage_meta(
    path: Path,
    scenario_id: str,
    config: Path,
    manifest: Path,
    stage: str,
    start: datetime,
    stop: datetime,
    error: str | None,
) -> None:
    meta = {
        "scenario_id": scenario_id,
        "stage": stage,
        "config_path": str(config),
        "config_sha256": _sha256_file(config),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "started_at_utc": start.isoformat(),
        "stopped_at_utc": stop.isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "error": error,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
