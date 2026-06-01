"""Prepare an immutable run directory config snapshot."""
from __future__ import annotations

import argparse
import os

import yaml

from pipeline.config.loader import load_config
from pipeline.config.sweep import (
    expand_parameter_sweep,
    parameter_sweep_enabled,
    probe5_ring_calibration_enabled,
)
from pipeline.manifest.io import atomic_write_json, init_manifest


def prepare_run(config_path: str, run_dir: str) -> tuple[list[str], str]:
    """Write run-local config, scenario index, manifest, and optional threshold stub."""
    cfg = load_config(config_path)
    expanded_cfg, scenario_names, threshold_stub = expand_parameter_sweep(cfg)

    os.makedirs(run_dir, exist_ok=True)
    run_config_path = os.path.join(run_dir, "config.yaml")
    with open(run_config_path, "w") as f:
        yaml.safe_dump(expanded_cfg, f, sort_keys=False)

    atomic_write_json(os.path.join(run_dir, "scenario_index.json"), scenario_names)
    init_manifest(run_dir, run_config_path, scenario_names)

    if probe5_ring_calibration_enabled(cfg):
        mode = "calibrated_parameter_sweep"
    elif parameter_sweep_enabled(cfg):
        mode = "parameter_sweep"
    else:
        mode = "threshold_search"
    if threshold_stub:
        atomic_write_json(os.path.join(run_dir, "threshold_result.json"), threshold_stub)

    return scenario_names, mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pump_multi_comparison run dir")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    scenario_names, mode = prepare_run(args.config, args.run_dir)
    print(f"mode={mode}")
    for name in scenario_names:
        print(name)


if __name__ == "__main__":
    main()
