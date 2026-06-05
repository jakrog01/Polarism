"""GPU stage: run all expanded spacetime mechanism scenarios sequentially."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

from mnist_common.io.atomic import atomic_write_json
from mnist_spacetime_v1.config.loader import load_config
from mnist_spacetime_v1.config.scenarios import expand_scenarios
from mnist_spacetime_v1.simulation.core import (
    SharedScenarioResources,
    build_polarism_config,
    run_scenario,
)
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters


def _ensure_run_dirs(run_dir: str) -> None:
    for name in ("traces", "metadata", "plots", "logs"):
        os.makedirs(os.path.join(run_dir, name), exist_ok=True)


def _scenario_index(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "name": scenario["name"],
            "architecture": scenario["architecture"],
            "pattern": scenario.get("pattern"),
            "n_lasers": len(scenario.get("lasers", [])),
            "n_rois": len(scenario.get("rois", [])),
            "metadata": scenario.get("metadata", {}),
        }
        for i, scenario in enumerate(scenarios)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mnist_spacetime_v1 scenario campaign")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"ERROR: config.yaml not found in run_dir: {run_dir}", file=sys.stderr)
        sys.exit(1)

    _ensure_run_dirs(run_dir)
    cfg = load_config(cfg_path)
    scenarios = expand_scenarios(cfg)

    atomic_write_json(os.path.join(run_dir, "scenario_index.json"), _scenario_index(scenarios))
    atomic_write_json(os.path.join(run_dir, "scenarios_expanded.json"), scenarios)

    print("=" * 70)
    print(" mnist_spacetime_v1 - GPU scenario campaign")
    print("=" * 70)
    print(f"  Run dir     : {run_dir}")
    print(f"  Config      : {cfg_path}")
    print(f"  Architecture: {cfg.get('architecture', {}).get('kind')}")
    print(f"  Scenarios   : {len(scenarios)}")
    print("")

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = build_polarism_config(cfg, use_gpu=True)
    resources = SharedScenarioResources(sim_cfg)

    manifest: dict[str, Any] = {
        "package": "mnist_spacetime_v1",
        "architecture": cfg.get("architecture", {}).get("kind"),
        "n_scenarios": len(scenarios),
        "scenario_names": [s["name"] for s in scenarios],
        "grid": cfg.get("global", {}).get("grid", {}),
        "solver": cfg.get("global", {}).get("solver", {}),
        "physics": cfg.get("global", {}).get("physics", {}),
        "output": cfg.get("output", {}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(os.path.join(run_dir, "manifest.json"), manifest)

    all_meta = []
    t0 = time.monotonic()
    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i:02d}/{len(scenarios):02d}] {scenario['name']}")
        meta = run_scenario(resources, scenario, cfg, run_dir)
        all_meta.append(meta)
        print(
            "    "
            f"condensed={meta['condensed']}  "
            f"t_cond={meta['t_cond_ps']}  "
            f"psi_peak={meta['psi_sq_max_peak']:.3e}  "
            f"elapsed={meta['elapsed_s']:.1f}s"
        )

    elapsed = time.monotonic() - t0
    atomic_write_json(
        os.path.join(run_dir, "campaign_gpu_summary.json"),
        {
            "n_scenarios": len(scenarios),
            "elapsed_s": round(elapsed, 2),
            "scenarios": [
                {
                    "name": m["scenario"]["name"],
                    "condensed": m["condensed"],
                    "t_cond_ps": m["t_cond_ps"],
                    "psi_sq_max_peak": m["psi_sq_max_peak"],
                    "elapsed_s": m["elapsed_s"],
                }
                for m in all_meta
            ],
        },
    )
    print("")
    print(f"  Done: {len(scenarios)} scenarios in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
