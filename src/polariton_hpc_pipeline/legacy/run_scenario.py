"""Legacy scenario runner."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import yaml

import simulate
import visualize
from config_loader import (
    get_laser_defaults,
    get_scenario,
    load_config,
    resolve_power,
)
from simulate import (
    RECORD_STRIDE,
    RNG_SEED,
    compute_batch_size,
    run_simulation,
)

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    ComputeEngineParameters,
    LaserParameters,
)
from polarism.grid.create_grid import create_grid
from polarism.laser.pulse_gaussian import PulseGaussian

if TYPE_CHECKING:
    from polarism.grid.simulation_grid_2d import SimulationGrid2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_original_build_base_config = simulate.build_base_config


def _patched_build_base_config(opt: dict[str, Any]) -> Any:
    """Patch the legacy base config with overrides."""
    cfg = _original_build_base_config(opt)
    for attr, key in [
        (cfg.physics, "_physics_overrides"),
        (cfg.boundary_condition, "_bc_overrides"),
        (cfg.reservoir, "_reservoir_overrides"),
        (cfg.solver, "_solver_overrides"),
        (cfg.grid, "_grid_overrides"),
        (cfg.potential, "_potential_overrides"),
    ]:
        for k, v in opt.get(key, {}).items():
            if hasattr(attr, k):
                setattr(attr, k, v)
    return cfg


def load_threshold_result() -> dict[str, Any]:
    """Load the saved threshold-search result."""
    path = os.path.join(SCRIPT_DIR, "threshold_result.json")
    with open(path) as f:
        return json.load(f)


def build_opt_dict(
    cfg: dict[str, Any],
    threshold: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Build the legacy option dict for one scenario."""
    g = cfg["global"]
    defaults = get_laser_defaults(cfg)
    return {
        "P_optimal": threshold["P_threshold"],
        "t_max": g["solver"]["total_time"],
        "dt": g["solver"]["dt"],
        "nx": g["grid"]["nx"],
        "ny": g["grid"]["ny"],
        "lx": g["grid"]["lx"],
        "ly": g["grid"]["ly"],
        "sigma_space": defaults.get(
            "sigma_space",
            threshold.get("sigma_space", 5.0),
        ),
        "sigma_time": threshold.get("sigma_time", 1.0),
        "pulse_separation": threshold.get("pulse_separation", 10.0),
        "cutoff_sigma": defaults.get(
            "cutoff_sigma",
            threshold.get("cutoff_sigma", 3.0),
        ),
        "_physics_overrides": g.get("physics", {}),
        "_bc_overrides": g.get("boundary_condition", {}),
        "_reservoir_overrides": g.get("reservoir", {}),
        "_solver_overrides": {
            "method": g.get("solver", {}).get("method", "rk4-cuda"),
        },
        "_grid_overrides": {
            "grid_type": g.get("grid", {}).get("grid_type", "periodic"),
        },
        "_potential_overrides": scenario.get("potential", {}),
    }


def build_scenario_lasers(
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    threshold: dict[str, Any],
    grid: SimulationGrid2D,
    rng: np.random.Generator,
) -> tuple[list[Any], list[float]]:
    """Build the lasers for one scenario."""
    defaults = get_laser_defaults(cfg)
    p_th: float = threshold["P_threshold"]
    th_sigma_time: float = threshold.get("sigma_time", 1.0)
    th_pulse_sep: float = threshold.get("pulse_separation", 10.0)

    laser_defs: list[dict[str, Any]] = scenario["lasers"]
    n_lasers = len(laser_defs)

    phases = rng.uniform(0, 2 * np.pi, size=n_lasers) if n_lasers > 1 else np.zeros(1)

    lasers: list[Any] = []
    for i, ldef in enumerate(laser_defs):
        merged = {**defaults, **ldef}
        power = resolve_power(merged.get("power"), p_th)
        sigma_time: float = merged.get("sigma_time", th_sigma_time)
        pulse_sep: float = merged.get("pulse_separation", th_pulse_sep)
        t_offset = float(phases[i] / (2 * np.pi) * pulse_sep) if n_lasers > 1 else 0.0

        laser_cfg = LaserParameters(
            mode="single",
            laser_type=merged.get("laser_type", "pulse-gaussian"),
            P0=power,
            Pmax=power,
            x0=merged.get("x0", 0.0),
            y0=merged.get("y0", 0.0),
            sigma_space=merged.get("sigma_space", 5.0),
            sigma_time=sigma_time,
            pulse_separation=pulse_sep,
            cutoff_sigma=merged.get("cutoff_sigma", 3.0),
            delay=t_offset,
        )
        lasers.append(PulseGaussian(laser_cfg, grid.X, grid.Y))

    return lasers, phases.tolist()


def setup_workdir() -> str:
    """Create a scratch work directory."""
    tmpdir = os.environ.get("TMPDIR")
    scratch = os.environ.get("SCRATCH")
    if tmpdir and os.path.isdir(tmpdir):
        d = tempfile.mkdtemp(prefix="polariton_run_", dir=tmpdir)
        print(f"  Using node-local scratch: {d}")
        return d
    if scratch and os.path.isdir(scratch):
        d = tempfile.mkdtemp(prefix="polariton_run_", dir=scratch)
        print(f"  Using scratch dir: {d}")
        return d
    print(f"  Using local directory: {SCRIPT_DIR}")
    return SCRIPT_DIR


def copy_back(work_dir: str, scenario_name: str) -> None:
    """Copy result files back from the work directory."""
    if work_dir == SCRIPT_DIR:
        return
    print(f"\n  Copying results back to {SCRIPT_DIR} ...")
    failed = False
    for name in [f"{scenario_name}.h5", "results_summary.json"]:
        src = os.path.join(work_dir, name)
        if not os.path.isfile(src):
            continue
        try:
            shutil.copy2(src, SCRIPT_DIR)
        except OSError as e:
            print(f"  ERROR copying {name}: {e}", file=sys.stderr)
            failed = True
    if failed:
        print(
            f"  WARNING: some copies failed. Keeping scratch: {work_dir}",
            file=sys.stderr,
        )
    else:
        print("  Done.")


def run_visualization(scenario_name: str, extent: list[float]) -> None:
    """Render the legacy scenario outputs."""
    visualize.DATA_DIR = SCRIPT_DIR
    visualize.RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

    out_dir = visualize.routine_dir(scenario_name)
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(SCRIPT_DIR, f"{scenario_name}.h5")
    if not os.path.isfile(h5_path):
        print(f"  Skipping visualisation: {h5_path} not found")
        return

    for field_key in visualize.FIELD_SPECS:
        print(f"    {field_key} ...", end="", flush=True)
        visualize.generate_field_png(scenario_name, field_key, extent)

    print("    animation ...", flush=True)
    visualize.generate_animation(scenario_name, extent)


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    laser_type = str(
        cfg.get("global", {}).get("laser_defaults", {}).get("laser_type", "pulse-gaussian")
    )
    if laser_type != "pulse-gaussian":
        print(
            f"ERROR: legacy runner does not support laser_type={laser_type!r}. "
            "Use the pipeline GPU stage (pipeline.stages.gpu.run_scenario) instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    threshold = load_threshold_result()

    if not threshold.get("search_completed"):
        print(
            "  ERROR: threshold search did not complete successfully.",
            file=sys.stderr,
        )
        sys.exit(1)

    scenario = get_scenario(cfg, args.scenario)

    print("=" * 60)
    print(f" Scenario: {scenario['name']}")
    print("=" * 60)
    print(f"  P_threshold : {threshold['P_threshold']:.1f}")

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    opt = build_opt_dict(cfg, threshold, scenario)

    simulate.build_base_config = _patched_build_base_config

    cfg_tmp = _patched_build_base_config(opt)
    grid = create_grid(cfg_tmp.grid)

    rng = np.random.default_rng(RNG_SEED)
    lasers, phases = build_scenario_lasers(
        scenario,
        cfg,
        threshold,
        grid,
        rng,
    )

    print(f"  Lasers: {len(lasers)}")
    for i, laser in enumerate(lasers):
        print(
            f"    [{i}] P={laser.P0:.2f}  "
            f"({laser.x0:.1f}, {laser.y0:.1f})  "
            f"t_off={laser.delay:.3f}"
        )

    batch_size = compute_batch_size(grid.ny, grid.nx)
    n_steps = int(opt["t_max"] / opt["dt"])
    est_frames = n_steps // RECORD_STRIDE
    print(f"  Batch: {batch_size},  Steps: {n_steps},  ~{est_frames} frames")

    work_dir = setup_workdir()

    try:
        print("\n  Running simulation ...")
        t_cond = run_simulation(
            scenario["name"],
            lasers,
            opt,
            batch_size,
            work_dir,
        )

        copy_back(work_dir, scenario["name"])

        logs_dir = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        scenario_cfg_data = {
            "scenario": scenario["name"],
            "P_threshold": threshold["P_threshold"],
            "lasers": [
                {
                    "x0": laser.x0,
                    "y0": laser.y0,
                    "P0": laser.P0,
                    "sigma_time": laser.sigma_time,
                    "pulse_separation": laser.pulse_separation,
                    "delay": laser.delay,
                }
                for laser in lasers
            ],
        }
        cfg_yaml_path = os.path.join(logs_dir, f"{scenario['name']}_config.yaml")
        with open(cfg_yaml_path, "w") as f:
            yaml.safe_dump(scenario_cfg_data, f, default_flow_style=False)

        summary_path = os.path.join(SCRIPT_DIR, "results_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary: dict[str, Any] = {
                "P_threshold": threshold["P_threshold"],
                "routines": {},
            }

        summary["routines"][scenario["name"]] = {
            "t_cond": t_cond,
            "file": f"{scenario['name']}.h5",
            "n_lasers": len(lasers),
            "phase_offsets": phases,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        lx: float = opt["lx"]
        ly: float = opt["ly"]
        extent = [-lx / 2, lx / 2, -ly / 2, ly / 2]

        print("\n  Generating visualisations ...")
        run_visualization(scenario["name"], extent)

    finally:
        if work_dir != SCRIPT_DIR and os.path.isdir(work_dir):
            try:
                shutil.rmtree(work_dir)
            except OSError:
                pass

    print(f"\n  Scenario '{scenario['name']}' complete.")


if __name__ == "__main__":
    main()
