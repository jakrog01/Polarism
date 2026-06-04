"""Calibration pilot: find P_ring_threshold and P_trigger_threshold for 16-spot ring.

Runs small sweeps (ring-only, trigger-only, ring+trigger assisted) and writes
calibration results to the run directory.

Invoked as:
    python -m mnist_ring_v1.stages.gpu.calibrate \\
        --run-dir <run_dir> [--mode ring|trigger|assisted|all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from mnist_ring_v1.encoding.geometry import ring_spot_positions
from mnist_ring_v1.simulation.batch_core import SharedSimResources, simulate_one_image
from mnist_ring_v1.simulation.lasers import build_ring_trigger_lasers
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    BoundaryConditionParameters,
    ComputeEngineParameters,
    Config,
    GridParameters,
    PhysicsConstants,
    PotentialParameters,
    ReservoirParameters,
    ResultParameters,
    SolverParameters,
)


def _atomic_write_json(path: str, data: Any) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _build_config(cfg: dict) -> Config:
    g = cfg["global"]["grid"]
    p = cfg["global"]["physics"]
    s = cfg["global"]["solver"]
    bc = cfg["global"].get("boundary_condition", {})
    r = cfg["global"].get("reservoir", {})
    return Config(
        grid=GridParameters(
            nx=int(g["nx"]),
            ny=int(g["ny"]),
            lx=float(g["lx"]),
            ly=float(g["ly"]),
            grid_type=str(g.get("grid_type", "periodic")),
        ),
        boundary_condition=BoundaryConditionParameters(
            profile_type=str(bc.get("profile_type", "sin2")),
            strength=float(bc.get("strength", 5.0)),
            absorption=str(bc.get("absorption", "cap")),
            mask_width_percent=float(bc.get("mask_width_percent", 0.2)),
        ),
        potential=PotentialParameters(potential_type="zero"),
        physics=PhysicsConstants(
            hbar=float(p["hbar"]),
            m_eff=float(p["m_eff"]),
            gamma_C=float(p["gamma_C"]),
            gamma_R=float(p["gamma_R"]),
            gamma_I=float(p.get("gamma_I", 0.001)),
            g_C=float(p["g_C"]),
            g_R=float(p["g_R"]),
            g_I=float(p.get("g_I", 0.0)),
            R=float(p["R"]),
            kappa=float(p.get("kappa", 0.05)),
            kinetic_relaxation_eta=float(p.get("kinetic_relaxation_eta", 1e-5)),
            init_mode=str(p.get("init_mode", "filtered_complex_gaussian")),
            init_k_cutoff_um=float(p.get("init_k_cutoff_um", 3.0)),
            init_seed=int(p.get("init_seed", 42)),
            init_eps=float(p.get("init_eps", 1e-3)),
        ),
        reservoir=ReservoirParameters(
            reservoir_type=str(r.get("reservoir_type", "quadratic-double")),
        ),
        solver=SolverParameters(
            total_time=float(s["total_time"]),
            dt=float(s["dt"]),
            method=str(s.get("method", "rk4-cuda")),
            laplacian=str(s.get("laplacian", "isotropic-9pt")),
        ),
        result=ResultParameters(real_time_view=False, save_results=False),
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )


def _sweep_threshold(
    resources: SharedSimResources,
    powers: list[float],
    make_lasers_fn,
    readout_start: float,
    readout_end: float,
    label: str,
    calib_stride: int = 500,
) -> dict[str, Any]:
    results = []
    for p in powers:
        lasers = make_lasers_fn(p)
        t0 = time.monotonic()
        result = simulate_one_image(
            resources=resources,
            lasers=lasers,
            readout_window_start_ps=readout_start,
            readout_window_end_ps=readout_end,
            kspace_crop_size=64,
            kspace_normalization="log_total_power",
            readout_stride_steps=calib_stride,
        )
        elapsed = time.monotonic() - t0
        results.append(
            {
                "power": p,
                "condensed": result["condensed"],
                "psi_sq_max": result["psi_sq_max"],
                "k0_frac": result["k0_frac"],
                "k_peak_um": result["k_peak_um"],
                "t_cond": result["t_cond"],
                "elapsed_s": round(elapsed, 2),
            }
        )
        marker = "COND" if result["condensed"] else "sub "
        print(
            f"    {label} P={p:.0f}: {marker}  psi_sq_max={result['psi_sq_max']:.3e}  t={elapsed:.1f}s"
        )

    bracket_found = False
    threshold = None
    for i in range(len(results) - 1):
        if not results[i]["condensed"] and results[i + 1]["condensed"]:
            threshold = 0.5 * (results[i]["power"] + results[i + 1]["power"])
            bracket_found = True
            break

    if not bracket_found:
        all_cond = all(r["condensed"] for r in results)
        none_cond = not any(r["condensed"] for r in results)
        if all_cond:
            print(
                f"  WARNING [{label}]: all {len(results)} points condensed — "
                f"threshold is below {results[0]['power']:.0f}. Widen sweep downward."
            )
        elif none_cond:
            print(
                f"  WARNING [{label}]: no points condensed — "
                f"threshold is above {results[-1]['power']:.0f}. Widen sweep upward."
            )

    return {
        "sweeps": results,
        "estimated_threshold": threshold,
        "bracket_found": bracket_found,
    }


def _select_guard_samples(
    dataset_index: list[dict],
    n_per_class: int = 2,
) -> list[dict]:
    """Return up to n_per_class entries per label, balanced across 10 classes."""
    from collections import defaultdict
    buckets: dict[int, list[dict]] = defaultdict(list)
    for entry in dataset_index:
        lbl = int(entry.get("label", -1))
        if 0 <= lbl <= 9 and len(buckets[lbl]) < n_per_class:
            buckets[lbl].append(entry)
    samples: list[dict] = []
    for lbl in sorted(buckets):
        samples.extend(buckets[lbl])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ring+trigger thresholds")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--mode", choices=["ring", "trigger", "assisted", "all"], default="all"
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    enc_cfg = cfg["encoding"]
    T_max = float(enc_cfg["T_max"])
    ring_r = float(enc_cfg["ring_radius_um"])
    ring_rot = float(enc_cfg["ring_rotation_rad"])
    sigma_space = float(enc_cfg["sigma_space_um"])
    sigma_time = float(enc_cfg["sigma_time_ps"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])

    total_time = float(cfg["global"]["solver"]["total_time"])
    readout_start = T_max + float(cfg["readout"]["window_start_offset_ps"])
    readout_end = T_max + float(cfg["readout"]["window_end_offset_ps"])
    calib_stride = int(cfg["readout"].get("calibration_stride_steps", 500))

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = _build_config(cfg)
    resources = SharedSimResources(sim_cfg)
    grid = resources.grid

    ring_xs, ring_ys = ring_spot_positions(ring_r, ring_rot)
    all_delays_mid = np.full(16, T_max * 0.5)
    trig_delay_val = T_max

    ring_powers = [2500, 3000, 3500, 4000, 5000, 6500, 8000, 10000]
    assisted_ring_powers = [1400, 1800, 2200, 2800, 3400, 4200, 5200]
    trigger_powers = [3000, 4000, 5000, 6000, 7000, 8000, 10000]

    results_all: dict[str, Any] = {}

    if args.mode in ("ring", "all"):
        print("\n=== Ring-only sweep (trigger OFF) ===")

        def make_ring_only(p: float):
            return build_ring_trigger_lasers(
                ring_xs=ring_xs,
                ring_ys=ring_ys,
                ring_delays=all_delays_mid,
                ring_power=p,
                trigger_delay=total_time + 100.0,
                trigger_power=0.0,
                sigma_space_um=sigma_space,
                sigma_time_ps=sigma_time,
                cutoff_sigma=cutoff,
                power_definition=power_def,
                grid_X=grid.X,
                grid_Y=grid.Y,
            )

        results_all["ring_only"] = _sweep_threshold(
            resources,
            ring_powers,
            make_ring_only,
            readout_start,
            readout_end,
            "ring",
            calib_stride,
        )

    if args.mode in ("trigger", "all"):
        print("\n=== Trigger-only sweep (ring OFF) ===")

        def make_trig_only(p: float):
            return build_ring_trigger_lasers(
                ring_xs=ring_xs,
                ring_ys=ring_ys,
                ring_delays=all_delays_mid,
                ring_power=0.0,
                trigger_delay=trig_delay_val,
                trigger_power=p,
                sigma_space_um=sigma_space,
                sigma_time_ps=sigma_time,
                cutoff_sigma=cutoff,
                power_definition=power_def,
                grid_X=grid.X,
                grid_Y=grid.Y,
            )

        results_all["trigger_only"] = _sweep_threshold(
            resources,
            trigger_powers,
            make_trig_only,
            readout_start,
            readout_end,
            "trig",
            calib_stride,
        )

    if args.mode in ("assisted", "all"):
        print("\n=== Assisted ring sweep (trigger=0.8*trigger_th) ===")
        trig_th = None
        if "trigger_only" in results_all:
            if results_all["trigger_only"].get("bracket_found"):
                trig_th = results_all["trigger_only"].get("estimated_threshold")
        if trig_th is None:
            print(
                "  WARNING: trigger threshold not bracketed; using fallback 6800. "
                "Fix trigger sweep first for meaningful assisted results."
            )
            trig_th = 6800.0
        assist_trig = 0.8 * trig_th
        print(f"  trigger fixed at {assist_trig:.0f}")

        def make_assisted(p: float):
            return build_ring_trigger_lasers(
                ring_xs=ring_xs,
                ring_ys=ring_ys,
                ring_delays=all_delays_mid,
                ring_power=p,
                trigger_delay=trig_delay_val,
                trigger_power=assist_trig,
                sigma_space_um=sigma_space,
                sigma_time_ps=sigma_time,
                cutoff_sigma=cutoff,
                power_definition=power_def,
                grid_X=grid.X,
                grid_Y=grid.Y,
            )

        results_all["assisted_ring"] = _sweep_threshold(
            resources,
            assisted_ring_powers,
            make_assisted,
            readout_start,
            readout_end,
            "asst",
            calib_stride,
        )
        results_all["assisted_ring"]["fixed_trigger_power"] = assist_trig

    _atomic_write_json(os.path.join(run_dir, "calibration_result.json"), results_all)

    required_brackets = {"ring_only", "trigger_only"}
    missing_brackets: list[str] = []
    for mode_key, res in results_all.items():
        th = res.get("estimated_threshold")
        bracket = res.get("bracket_found", False)
        status = "OK" if bracket else "NO BRACKET — widen sweep"
        print(
            f"  {mode_key}: estimated_threshold={th}  bracket_found={bracket}  [{status}]"
        )
        if mode_key in required_brackets and not bracket:
            missing_brackets.append(mode_key)

    if missing_brackets:
        print(
            f"\nERROR: Required thresholds have no bracket: {missing_brackets}\n"
            "Widen the power sweep and rerun calibration.",
            file=sys.stderr,
        )
        sys.exit(1)

    ring_threshold = results_all["ring_only"]["estimated_threshold"]
    if ring_threshold is None:
        print("ERROR: ring_only threshold is None after bracket check.", file=sys.stderr)
        sys.exit(1)

    # ── Pre-trigger guard ─────────────────────────────────────────────────────
    # Run ring-only with real TTFS delays from dataset_index.json.
    # Simulate to readout_end (T_max + 35 ps = 135 ps) to check that ring alone
    # causes no central condensation anywhere in the trigger+readout window.
    # Using T_max+5 (105 ps) would miss ring-only central condensation during
    # 110-135 ps; extending to readout_end adds ~8 s/img (135 vs 105 ps)
    # and gives a physically complete guard.
    print("\n=== Pre-trigger guard (ring-only, real TTFS delays) ===")
    dataset_index_path = os.path.join(run_dir, "dataset_index.json")
    if not os.path.isfile(dataset_index_path):
        print(
            "  WARNING: dataset_index.json not found; skipping pre-trigger guard.\n"
            "  Run prepare_run first so calibrate has real TTFS delays to test."
        )
        results_all["pretrigger_guard"] = {"skipped": True, "reason": "no dataset_index.json"}
    else:
        with open(dataset_index_path) as _f:
            dataset_index = json.load(_f)

        guard_samples = _select_guard_samples(dataset_index, n_per_class=2)
        print(f"  Guard samples: {len(guard_samples)} images  (2/class balanced)")

        guard_stop_ps = readout_end  # T_max + 35 ps = 135 ps
        guard_fractions = [0.35, 0.45, 0.55, 0.65, 0.75, 0.80]
        central_roi_um = float(cfg.get("readout", {}).get("central_roi_um", 6.0))

        guard_results: list[dict] = []
        for frac in guard_fractions:
            ring_power_guard = frac * ring_threshold
            n_pretrigger = 0
            for entry in guard_samples:
                ring_delays_real = np.array(entry["ring_delays_ps"], dtype=np.float64)
                lasers_guard = build_ring_trigger_lasers(
                    ring_xs=ring_xs,
                    ring_ys=ring_ys,
                    ring_delays=ring_delays_real,
                    ring_power=ring_power_guard,
                    trigger_delay=total_time + 100.0,
                    trigger_power=0.0,
                    sigma_space_um=sigma_space,
                    sigma_time_ps=sigma_time,
                    cutoff_sigma=cutoff,
                    power_definition=power_def,
                    grid_X=grid.X,
                    grid_Y=grid.Y,
                )
                result = simulate_one_image(
                    resources=resources,
                    lasers=lasers_guard,
                    readout_window_start_ps=guard_stop_ps,
                    readout_window_end_ps=guard_stop_ps,
                    kspace_crop_size=16,
                    kspace_normalization="log_total_power",
                    readout_stride_steps=calib_stride,
                    trigger_delay_ps=T_max,
                    central_roi_um=central_roi_um,
                    stop_time_ps=guard_stop_ps,
                )
                # Fail if ring-only causes ANY central condensation up to guard_stop_ps.
                # trigger is OFF here, so central condensation at any time is a problem.
                if result["central_condensed"]:
                    n_pretrigger += 1

            safe = n_pretrigger == 0
            guard_results.append({
                "fraction": frac,
                "ring_power": ring_power_guard,
                "n_samples": len(guard_samples),
                "n_central_condensed": n_pretrigger,
                "safe": safe,
            })
            marker = "SAFE" if safe else f"FAIL ({n_pretrigger}/{len(guard_samples)} central-condensed)"
            print(
                f"  frac={frac:.2f}  P_ring={ring_power_guard:.0f}  [{marker}]"
            )

        safe_fracs = [r["fraction"] for r in guard_results if r["safe"]]
        recommended_fraction = max(safe_fracs) if safe_fracs else None

        results_all["pretrigger_guard"] = {
            "ring_only_threshold": ring_threshold,
            "guard_stop_ps": guard_stop_ps,
            "central_roi_um": central_roi_um,
            "n_guard_samples": len(guard_samples),
            "fractions_tested": guard_results,
            "recommended_ring_fraction": recommended_fraction,
            "safe_fractions": safe_fracs,
            "skipped": False,
        }

        if recommended_fraction is None:
            print(
                "\nERROR: No safe ring fraction found — all tested fractions cause "
                "pre-trigger central condensation.\n"
                "Consider reducing ring_only power range or verifying sigma_space.",
                file=sys.stderr,
            )
            _atomic_write_json(os.path.join(run_dir, "calibration_result.json"), results_all)
            sys.exit(1)

        print(f"\n  recommended_ring_fraction = {recommended_fraction:.2f}  "
              f"(P_ring = {recommended_fraction * ring_threshold:.0f})")

    out = os.path.join(run_dir, "calibration_result.json")
    _atomic_write_json(out, results_all)
    print(f"\nCalibration written: {out}")


if __name__ == "__main__":
    main()
