"""Finalize ablation metrics for one campaign scenario."""
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

from mnist_digits_polariton_snn_dynamic.config.loader import load_snn_dynamic_config
from mnist_digits_polariton_snn_dynamic.readout.ablation import (
    aggregate_campaign_ablation,
    finalize_ablation,
)
from mnist_digits_polariton_snn_dynamic.readout.heatmaps import generate_trace_heatmaps


CALIBRATION_CAVEAT = (
    "Threshold calibration for this scenario measured full-lattice, central "
    "single-spot, and p95/p99 representative encoded-training-image thresholds. "
    "The gap between synthetic extremes is therefore measured directly on "
    "heterogeneous training-image powers, not inferred from two boundary points. "
    "final_power_max uses the full-lattice margin and caps from the single-spot "
    "and realistic-image thresholds; no baseline threshold multiplier is used. "
    "Representative realistic images are selected by a clustered-neighborhood "
    "power metric, then the worst, lowest-threshold case from top_k candidates "
    "is used. "
    "Differences from pitch12_sigma2_baseline should be interpreted under each "
    "scenario's own geometry-calibrated final_power_max."
)


def main() -> None:
    """Run CPU-only ablation finalize for one campaign scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaign-output-dir", required=True)
    args = parser.parse_args()
    finalize_scenario(args.manifest, args.scenario_id, args.campaign_output_dir)


def finalize_scenario(
    manifest_path: str,
    scenario_id: str,
    campaign_output_dir: str,
) -> None:
    """Finalize one scenario and refresh campaign-level ablation summary."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_data = _load_yaml(manifest)
    scenario = _find_scenario(manifest_data, scenario_id)
    config = (manifest.parent / str(scenario["config"])).resolve()
    campaign_dir = Path(campaign_output_dir).expanduser().resolve()
    output_dir = campaign_dir / scenario_id
    skip_marker = output_dir / "skipped.json"
    if skip_marker.exists():
        reason = _read_skip_reason(skip_marker)
        print(
            f"SKIP: scenario {scenario_id!r} finalize stage skipped: {reason}",
            file=sys.stderr,
            flush=True,
        )
        now = datetime.now(timezone.utc)
        _write_stage_meta(
            output_dir / "finalize_meta.json",
            scenario_id,
            config,
            manifest,
            "finalize",
            now,
            now,
            error=None,
            skipped=True,
            skip_reason=reason,
        )
        try:
            aggregate_campaign_ablation(str(campaign_dir))
        except Exception as exc:
            print(
                f"WARNING: aggregate_campaign_ablation failed for skipped scenario "
                f"{scenario_id!r}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        return
    start = datetime.now(timezone.utc)
    error: str | None = None
    try:
        baseline_scenario_id = _baseline_scenario_id(manifest_data)
        caveat = None if scenario_id == baseline_scenario_id else CALIBRATION_CAVEAT
        finalize_ablation(scenario_id, str(config), str(output_dir), caveat)
        snn_cfg = load_snn_dynamic_config(str(config))
        generate_trace_heatmaps(output_dir, n_side=snn_cfg.geometry.n_side)
        aggregate_campaign_ablation(str(campaign_dir))
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        stop = datetime.now(timezone.utc)
        _write_stage_meta(
            output_dir / "finalize_meta.json",
            scenario_id,
            config,
            manifest,
            "finalize",
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


def _baseline_scenario_id(manifest_data: dict[str, Any]) -> str:
    scenarios = manifest_data.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("manifest.scenarios must be a non-empty list")
    baseline = manifest_data.get("baseline_scenario_id")
    if baseline is not None:
        return str(baseline)
    first = scenarios[0]
    if not isinstance(first, dict) or "id" not in first:
        raise ValueError("First scenario must define id when baseline_scenario_id is absent")
    return str(first["id"])


def _read_skip_reason(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception:
        return None
    reason = data.get("reason") if isinstance(data, dict) else None
    return str(reason) if reason is not None else None


def _write_stage_meta(
    path: Path,
    scenario_id: str,
    config: Path,
    manifest: Path,
    stage: str,
    start: datetime,
    stop: datetime,
    error: str | None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
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
