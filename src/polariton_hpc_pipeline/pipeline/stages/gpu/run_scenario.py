"""GPU stage: scenario simulation + scratch-local rendering.

Runs one scenario (identified by ``$SLURM_ARRAY_TASK_ID`` or
``--scenario-index``), renders PNGs and an animation from the HDF5 while it
is still on local scratch, then persists only lightweight artifacts to the
run directory on ``/lu/tetyda``.  Raw HDF5 is copied to the run directory
only when ``archive_raw_hdf5: true`` is set in the config ``output`` section.

Invoked by Slurm as:
    python -m pipeline.stages.gpu.run_scenario --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any
import warnings

import numpy as np

from pipeline.config.builder import (
    build_scenario_config,
    build_scenario_lasers,
    build_scenario_rois,
)
from pipeline.config.loader import get_scenario, load_config
from pipeline.config.output_policy import output_policy_from_config
from pipeline.manifest.io import (
    atomic_write_json,
    resolve_scenario_name,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.render.nvenc_stream import generate_animation
from pipeline.simulation.core import RNG_SEED, run_simulation_from_config
from polarism.solver.solver_compatibility import (
    SolverCompatibilityError,
    check_solver_compatibility,
)
from pipeline.stages.cpu.viz_engine import (
    FIELD_SPECS,
    generate_field_png,
    generate_scalar_traces,
)
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.env_metadata import attach_environment
from polarism.grid.create_grid import create_grid


def _make_scratch_dir(run_dir: str, scenario_name: str) -> str:
    """Return a job-unique temp directory.

    Prefers ``$SCRATCH`` over Slurm/node-local temp directories because HDF5
    outputs can be large.
    """
    run_name = os.path.basename(run_dir.rstrip("/"))
    job_id = (
        os.environ.get("POLARITON_SCRATCH_ID")
        or os.environ.get("SLURM_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or "local"
    )
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    unique_id = f"{job_id}_{task_id}"

    for env_var in ("SCRATCH", "SLURM_TMPDIR", "TMPDIR"):
        base = os.environ.get(env_var)
        if base and os.path.isdir(base):
            if os.path.realpath(base) == "/tmp":
                continue
            scratch = os.path.join(base, "polariton", run_name, unique_id, scenario_name)
            os.makedirs(scratch, exist_ok=False)
            return scratch

    if os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError(
            "No suitable scratch directory found for the GPU scenario job. "
            "Set and export SCRATCH in slurm.env or provide SLURM_TMPDIR/TMPDIR "
            "that points to real job scratch, not /tmp."
        )

    return run_dir


def _effective_sweep_meta(
    scenario: dict[str, Any],
    lasers: list[Any],
    laser_ids: list[str],
) -> dict[str, Any] | None:
    sweep = scenario.get("sweep")
    if not isinstance(sweep, dict):
        return sweep

    out = copy.deepcopy(sweep)
    if out.get("mode") != "probe5":
        return out

    ring_powers = [float(laser.P0) for laser in lasers[:4]]
    if ring_powers:
        out["ring_energy"] = ring_powers[0]
        out["ring_energies"] = ring_powers

    probe_laser_id = str(out.get("probe_laser_id", "probe"))
    for i, laser_id in enumerate(laser_ids):
        if laser_id == probe_laser_id and i < len(lasers):
            out["probe_energy"] = float(lasers[i].P0)
            break

    return out


def _compute_laser_centering(
    sim_cfg: "Config",
    lasers: list,
    laser_ids: list[str],
) -> dict:
    """Return grid centering diagnostics for each laser.

    Computes nearest grid node and sub-cell offset for each laser's (x0, y0)
    without touching CuPy arrays — uses grid config parameters only.
    """
    from polarism.config.simulation_parameters import Config as _Config

    gc = sim_cfg.grid
    nx, ny = int(gc.nx), int(gc.ny)
    lx, ly = float(gc.lx), float(gc.ly)
    grid_type = str(getattr(gc, "grid_type", "periodic"))

    if grid_type == "closed-interval":
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)
        has_origin_x = (nx % 2 == 1)
        has_origin_y = (ny % 2 == 1)

        def _nearest(val: float, n: int, length: float, d: float) -> float:
            i = round((val + length / 2) / d)
            i = max(0, min(n - 1, i))
            return -length / 2 + i * d
    else:
        dx = lx / nx
        dy = ly / ny
        has_origin_x = (nx % 2 == 1)
        has_origin_y = (ny % 2 == 1)

        def _nearest(val: float, n: int, length: float, d: float) -> float:
            i = round(val / d + (n - 1) * 0.5)
            i = max(0, min(n - 1, i))
            return (i - (n - 1) * 0.5) * d

    laser_centering = []
    for lid, laser in zip(laser_ids, lasers):
        x0, y0 = float(laser.x0), float(laser.y0)
        nx_node = _nearest(x0, nx, lx, dx)
        ny_node = _nearest(y0, ny, ly, dy)
        ox = x0 - nx_node
        oy = y0 - ny_node
        laser_centering.append({
            "id": lid,
            "x0": x0,
            "y0": y0,
            "nearest_grid_x": round(nx_node, 10),
            "nearest_grid_y": round(ny_node, 10),
            "offset_from_nearest_grid_x": round(ox, 10),
            "offset_from_nearest_grid_y": round(oy, 10),
            "offset_in_dx": round(ox / dx, 6),
            "offset_in_dy": round(oy / dy, 6),
        })

    return {
        "dx": dx,
        "dy": dy,
        "has_origin_node_x": has_origin_x,
        "has_origin_node_y": has_origin_y,
        "laser_centering": laser_centering,
    }


def _laser_timing_end(laser: Any) -> float:
    sigma_time = getattr(laser, "sigma_time", None)
    if sigma_time is None:
        return laser.delay
    cutoff = float(getattr(laser, "cutoff_sigma", 3.0))
    n_pulses = int(getattr(laser, "n_pulses", 0))
    if n_pulses > 0:
        return (
            laser.delay
            + (n_pulses - 1) * float(laser.pulse_separation)
            + 2.0 * cutoff * float(sigma_time)
        )
    return laser.delay + 2.0 * cutoff * float(sigma_time)


_BASE_ANIMATION_PANELS = [
    ("psi",  "magma",   "abs2",  None,    1.0),
    ("nA",   "inferno", None,    None,    1.0),
    ("nI",   "magma",   None,    None,    1.0),
]


def _build_animation_field_specs(
    clim_overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict]:
    """Return movie panel specs for post-simulation HDF5 rendering."""
    overrides = clim_overrides or {}
    specs: dict[str, dict] = {}
    for src, cmap, transform, norm, gamma in _BASE_ANIMATION_PANELS:
        spec: dict = {
            "source": src,
            "cmap": cmap,
            "transform": transform,
            "norm": norm,
            "norm_gamma": gamma,
            "zero_vmin": True,
        }
        clim = overrides.get(src)
        if clim is None and src == "psi":
            clim = overrides.get("psi_sq")
        if clim is not None:
            spec["clim"] = clim
        specs[src] = spec
    return specs


def _render_animation_artifact(
    scenario_name: str,
    data_dir: str,
    scratch_results_dir: str,
    extent: list[float],
    clim_overrides: dict[str, tuple[float, float]] | None = None,
    downscale_factor: int = 1,
) -> None:
    """Generate the movie from scratch-local HDF5 after the run completes.

    The renderer first scans all recorded HDF5 frames to obtain global colour
    limits.  This keeps low-density regions black and prevents late condensate
    frames from saturating because of early-frame autoscaling.
    """
    field_specs = _build_animation_field_specs(clim_overrides)
    generate_animation(
        scenario_name,
        field_specs,
        extent,
        data_dir=data_dir,
        results_dir=scratch_results_dir,
        downscale_factor=downscale_factor,
    )


def _render_png_artifacts(
    scenario_name: str,
    data_dir: str,
    scratch_results_dir: str,
    extent: list[float],
    clim_overrides: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Generate PNG snapshot panels from scratch-local HDF5."""
    overrides = clim_overrides or {}
    for field_key in FIELD_SPECS:
        print(f"  PNG {field_key} ...", end="", flush=True)
        clim = overrides.get(field_key)
        try:
            generate_field_png(
                scenario_name,
                field_key,
                extent,
                data_dir,
                scratch_results_dir,
                clim=clim,
            )
            print(" done")
        except Exception as e:
            print(f" WARNING: {e}", file=sys.stderr)


def _copyback_artifacts(
    scratch_dir: str,
    run_dir: str,
    scenario_name: str,
    archive_raw_hdf5: bool,
) -> float:
    """Copy lightweight artifacts from scratch to run_dir; return seconds elapsed."""
    t0 = time.monotonic()

    src_scenario_results = os.path.join(scratch_dir, "results", scenario_name)
    dst_scenario_results = os.path.join(run_dir, "results", scenario_name)
    if os.path.isdir(src_scenario_results):
        os.makedirs(os.path.dirname(dst_scenario_results), exist_ok=True)
        if os.path.isdir(dst_scenario_results):
            shutil.rmtree(dst_scenario_results)
        shutil.copytree(src_scenario_results, dst_scenario_results)

    src_sidecar = os.path.join(scratch_dir, f"{scenario_name}_scalars.npz")
    if os.path.isfile(src_sidecar):
        shutil.copy2(src_sidecar, os.path.join(run_dir, f"{scenario_name}_scalars.npz"))

    src_fringe = os.path.join(scratch_dir, f"{scenario_name}_fringe.json")
    if os.path.isfile(src_fringe):
        shutil.copy2(src_fringe, os.path.join(run_dir, f"{scenario_name}_fringe.json"))

    if archive_raw_hdf5:
        src_h5 = os.path.join(scratch_dir, f"{scenario_name}.h5")
        if os.path.isfile(src_h5):
            dst_h5 = os.path.join(run_dir, f"{scenario_name}.h5")
            tmp_dst = dst_h5 + ".tmp"
            try:
                shutil.copy2(src_h5, tmp_dst)
                os.replace(tmp_dst, dst_h5)
            except Exception:
                traceback.print_exc()
                if os.path.isfile(tmp_dst):
                    try:
                        os.remove(tmp_dst)
                    except OSError:
                        pass
                raise RuntimeError(
                    f"HDF5 archive copy failed: {src_h5} -> {dst_h5}"
                )

    return time.monotonic() - t0


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="GPU scenario simulation + render (array task)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--scenario-index", type=int, default=None,
        help="0-based index into scenario_index.json.  "
             "Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    if args.scenario_index is not None:
        scenario_index = args.scenario_index
    else:
        slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task is None:
            print(
                "ERROR: --scenario-index not given and SLURM_ARRAY_TASK_ID not set.",
                file=sys.stderr,
            )
            sys.exit(1)
        scenario_index = int(slurm_task)

    try:
        scenario_name = resolve_scenario_name(run_dir, scenario_index)
    except (FileNotFoundError, IndexError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    config_path = os.path.join(run_dir, "config.yaml")
    threshold_path = os.path.join(run_dir, "threshold_result.json")
    for path in (config_path, threshold_path):
        if not os.path.isfile(path):
            print(f"ERROR: required file missing: {path}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config(config_path)
    with open(threshold_path) as f:
        threshold: dict = json.load(f)

    if not threshold.get("search_completed"):
        print("ERROR: threshold search did not complete successfully.", file=sys.stderr)
        sys.exit(1)

    output_policy = output_policy_from_config(cfg)
    global_cfg = cfg["global"]
    scenario = get_scenario(cfg, scenario_name)

    print("=" * 60)
    print(f" GPU Scenario: {scenario_name}  (index {scenario_index})")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    _is_sweep = threshold.get("mode") == "parameter_sweep"
    if _is_sweep:
        print(f"  P_threshold : {threshold['P_threshold']:.1f}  (stub — effective P set per laser)")
    else:
        print(f"  P_threshold : {threshold['P_threshold']:.1f}")
    print(f"  archive_h5  : {output_policy.archive_raw_hdf5}")
    print(
        f"  SLURM job   : "
        f"{os.environ.get('SLURM_JOB_ID', 'N/A')} / "
        f"task {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}"
    )

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))

    sim_cfg = build_scenario_config(global_cfg, threshold, scenario)

    try:
        with warnings.catch_warnings(record=True) as _compat_warns:
            warnings.simplefilter("always")
            check_solver_compatibility(sim_cfg)
        for w in _compat_warns:
            print(f"  SOLVER WARNING: {w.message}", file=sys.stderr)
    except SolverCompatibilityError as exc:
        print(f"  SOLVER COMPATIBILITY ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    grid = create_grid(sim_cfg.grid)
    extent = [
        -sim_cfg.grid.lx / 2,
        sim_cfg.grid.lx / 2,
        -sim_cfg.grid.ly / 2,
        sim_cfg.grid.ly / 2,
    ]

    rng = np.random.default_rng(RNG_SEED)
    lasers, phases = build_scenario_lasers(scenario, global_cfg, threshold, grid, rng)
    roi_specs = build_scenario_rois(
        scenario,
        global_cfg,
        threshold,
        lasers,
        output_cfg=cfg.get("output", {}),
    )

    print(f"  Lasers: {len(lasers)}")
    for i, laser in enumerate(lasers):
        print(
            f"    [{i}] {laser.laser_type} ({type(laser).__name__})  P={laser.P0:.2f}  "
            f"pos=({laser.x0:.1f}, {laser.y0:.1f})  "
            f"delay={laser.delay:.3f} ps"
        )
    if roi_specs:
        print("  ROIs:")
        for roi in roi_specs:
            print(
                f"    {roi['id']}: circle center=({roi['x0']:.3g}, {roi['y0']:.3g}) "
                f"r={roi['radius']:.3g}"
            )

    required_time = max(_laser_timing_end(laser) for laser in lasers)
    if required_time > sim_cfg.solver.total_time:
        print(
            "ERROR: scenario timing does not fit inside the configured simulation "
            f"window. Need at least {required_time:.1f} ps, but total_time="
            f"{sim_cfg.solver.total_time:.1f} ps.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_steps = int(sim_cfg.solver.total_time / sim_cfg.solver.dt)
    print(f"  Steps: {n_steps:,}")

    try:
        scratch_dir = _make_scratch_dir(run_dir, scenario_name)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    using_scratch = scratch_dir != run_dir
    if using_scratch:
        print(f"  Scratch: {scratch_dir}")

    scratch_results_dir = os.path.join(scratch_dir, "results")

    try:
        print("\n  Running simulation ...")
        t_sim_start = time.monotonic()
        t_cond, sidecar_path, ic_meta = run_simulation_from_config(
            scenario_name, lasers, sim_cfg, scratch_dir, output_policy,
            roi_specs=roi_specs, animation_visitor=None,
        )
        elapsed_sim = time.monotonic() - t_sim_start
    except Exception:
        traceback.print_exc()
        if using_scratch and os.path.isdir(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)
        sys.exit(1)

    animation_status = "disabled"
    animation_error: str | None = None

    print("\n  Rendering PNG artifacts from scratch ...")
    try:
        print("  scalar traces ...", end="", flush=True)
        generate_scalar_traces(scenario_name, scratch_dir, scratch_results_dir)
        print(" done")
    except Exception as e:
        print(f" WARNING: {e}", file=sys.stderr)
    if output_policy.render_snapshots:
        _render_png_artifacts(
            scenario_name,
            scratch_dir,
            scratch_results_dir,
            extent,
            clim_overrides=output_policy.animation_clim or None,
        )
    if output_policy.render_animation:
        print("  animation ...", flush=True)
        try:
            _render_animation_artifact(
                scenario_name,
                scratch_dir,
                scratch_results_dir,
                extent,
                clim_overrides=output_policy.animation_clim or None,
                downscale_factor=output_policy.render_downscale_factor,
            )
            animation_status = "ok"
        except Exception as e:
            animation_status = "failed"
            animation_error = str(e)
            traceback.print_exc()
            print(f" WARNING: animation render failed: {e}", file=sys.stderr)

    fringe_sidecar_written = False
    if output_policy.fringe_analysis_enabled:
        print("\n  Fringe analysis ...", end="", flush=True)
        try:
            from pipeline.analysis.fringe import compute_fringe_sidecar

            pump_cores = [(float(laser.x0), float(laser.y0)) for laser in lasers]
            pump_core_excl = (
                float(getattr(lasers[0], "cutoff_sigma", 3.0))
                * float(getattr(lasers[0], "sigma_space", 2.0))
                if lasers else 6.0
            )
            pump_end_t = max((_laser_timing_end(laser) for laser in lasers), default=0.0)
            fringe_data = compute_fringe_sidecar(
                scenario_name,
                scratch_dir,
                output_policy.fringe_analysis_window_radius,
                sim_cfg,
                threshold_criterion=5e-2,
                pump_cores=pump_cores,
                pump_core_exclusion_radius=pump_core_excl,
                t_min_ps=pump_end_t + 1.0,
            )
            fringe_path = os.path.join(scratch_dir, f"{scenario_name}_fringe.json")
            atomic_write_json(fringe_path, fringe_data)
            fringe_sidecar_written = True
            print(" done")
        except Exception as e:
            print(f" WARNING: fringe analysis failed: {e}", file=sys.stderr)
            traceback.print_exc()

    copyback_seconds: float = 0.0
    if using_scratch:
        print(f"\n  Copying artifacts scratch → {run_dir} ...")
        try:
            copyback_seconds = _copyback_artifacts(
                scratch_dir, run_dir, scenario_name, output_policy.archive_raw_hdf5
            )
        except Exception:
            traceback.print_exc()
            print(
                "ERROR: artifact copy-back failed; preserving scratch for recovery: "
                f"{scratch_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"  Copy-back done  ({copyback_seconds:.1f}s)")
        shutil.rmtree(scratch_dir, ignore_errors=True)

    h5_final = os.path.join(run_dir, f"{scenario_name}.h5")
    h5_bytes = os.path.getsize(h5_final) if os.path.isfile(h5_final) else 0

    laser_ids = [
        str(ld.get("id", f"laser_{i}"))
        for i, ld in enumerate(scenario.get("lasers", []))
    ]
    centering = _compute_laser_centering(sim_cfg, lasers, laser_ids)
    sweep_meta = _effective_sweep_meta(scenario, lasers, laser_ids)

    first_laser_power_def = (
        getattr(lasers[0], "power_definition", "peak_amplitude") if lasers else "peak_amplitude"
    )
    meta: dict = {
        "scenario": scenario_name,
        "scenario_index": scenario_index,
        "initial_condition": ic_meta,
        "P_threshold": threshold["P_threshold"],
        "effective_power": float(lasers[0].P0) if lasers else None,
        "effective_power_definition": first_laser_power_def,
        "t_cond": t_cond,
        "h5_file": f"{scenario_name}.h5" if output_policy.archive_raw_hdf5 else None,
        "sidecar_file": f"{scenario_name}_scalars.npz",
        "fringe_sidecar_file": (
            f"{scenario_name}_fringe.json" if fringe_sidecar_written else None
        ),
        "n_lasers": len(lasers),
        "phase_offsets": phases,
        "sweep": sweep_meta,
        "grid": {
            "nx": int(sim_cfg.grid.nx),
            "ny": int(sim_cfg.grid.ny),
            "lx": float(sim_cfg.grid.lx),
            "ly": float(sim_cfg.grid.ly),
            "grid_type": str(sim_cfg.grid.grid_type),
            "dx": centering["dx"],
            "dy": centering["dy"],
            "has_origin_node_x": centering["has_origin_node_x"],
            "has_origin_node_y": centering["has_origin_node_y"],
        },
        "rois": roi_specs,
        "lasers": [
            {
                "id": laser_ids[i] if i < len(laser_ids) else f"laser_{i}",
                "laser_type": laser.laser_type,
                "runtime_class": type(laser).__name__,
                "x0": float(laser.x0), "y0": float(laser.y0),
                "P0": float(laser.P0),
                "input_power": float(laser.P0),
                "power_definition": getattr(laser, "power_definition", "peak_amplitude"),
                **(
                    {
                        "pulse_energy": float(laser.P0),
                        "peak_amplitude_at_center": float(laser.peak_amplitude_at_center),
                        "space_integral": float(laser.spatial_integral),
                        "time_integral": float(laser.temporal_integral),
                    }
                    if getattr(laser, "power_definition", "peak_amplitude") == "pulse_energy"
                    else {}
                ),
                **(
                    {
                        "sigma_time": float(laser.sigma_time),
                        "pulse_separation": float(laser.pulse_separation),
                        "n_pulses": int(getattr(laser, "n_pulses", 0)),
                    }
                    if getattr(laser, "sigma_time", None) is not None
                    else {}
                ),
                "sigma_space": float(laser.sigma_space),
                "delay": float(laser.delay),
                **{k: v for k, v in centering["laser_centering"][i].items()
                   if k not in ("id", "x0", "y0")},
            }
            for i, laser in enumerate(lasers)
        ],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "animation_status": animation_status,
        "animation_error": animation_error,
        "telemetry": {
            "elapsed_sim_seconds": round(elapsed_sim, 2),
            "n_steps": n_steps,
            "h5_bytes": h5_bytes,
            "copyback_seconds": round(copyback_seconds, 2),
        },
    }
    atomic_write_json(scenario_meta_path(run_dir, scenario_name), attach_environment(meta))
    print(f"  Metadata: {scenario_meta_path(run_dir, scenario_name)}")
    print(f"\n  Scenario '{scenario_name}' complete. t_cond={t_cond}")

    if output_policy.render_animation and animation_status != "ok":
        print(
            f"ERROR: animation was required but status={animation_status!r}. "
            f"Simulation artifacts (HDF5, sidecar, PNGs) are preserved. "
            f"Error: {animation_error}",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
