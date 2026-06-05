"""Calibrate single-spot threshold and condensation threshold for WTA.

For each geometry radius, runs a power sweep with a single class spot
to find P_th_single_spot. Also runs blank (equal-score) images to
calibrate threshold_cond.

Invoked as:
    python -m mnist_wta_v1.stages.gpu.calibrate_wta \\
        --run-dir <run_dir> [--radius 12.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import yaml

from mnist_wta_v1.config.loader import (
    get_calibration_cfg,
    get_encoding_cfg,
    get_geometry_cfg,
    get_readout_cfg,
)
from mnist_wta_v1.encoding.class_geometry import class_spot_positions
from mnist_wta_v1.encoding.score_encoding import scores_to_powers
from mnist_wta_v1.simulation.batch_core import SharedSimResources, simulate_one_image_wta
from mnist_wta_v1.simulation.class_lasers import build_class_lasers
from mnist_common.io.atomic import atomic_write_json
from mnist_common.simulation.roi import build_class_roi_masks
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


def _run_single_spot_sweep(
    resources: SharedSimResources,
    x_spot: float,
    y_spot: float,
    powers: list[float],
    enc_cfg: dict,
    readout_cfg: dict,
) -> dict[str, Any]:
    sigma_space = float(enc_cfg["sigma_space_um"])
    sigma_time = float(enc_cfg["sigma_time_ps"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])
    pulse_timing_mode = str(enc_cfg.get("pulse_timing_mode", "simultaneous"))
    min_gap = float(enc_cfg.get("min_inter_pulse_gap_ps", 0.0))
    readout_start = float(readout_cfg["window_start_ps"])
    readout_end = float(readout_cfg["window_end_ps"])
    roi_r = float(readout_cfg["roi_radius_um"])
    calib_stride = int(readout_cfg.get("calibration_stride_steps", 500))
    grid = resources.grid
    xp = compute_engine.xp

    roi_masks = [
        xp.array(
            ((grid.X - x_spot) ** 2 + (grid.Y - y_spot) ** 2) <= roi_r ** 2
        )
    ]

    results = []
    for p in powers:
        lasers = build_class_lasers(
            xs_um=np.array([x_spot]),
            ys_um=np.array([y_spot]),
            powers=np.array([p]),
            sigma_space_um=sigma_space,
            sigma_time_ps=sigma_time,
            cutoff_sigma=cutoff,
            power_definition=power_def,
            grid_X=grid.X,
            grid_Y=grid.Y,
            pulse_timing_mode=pulse_timing_mode,
            min_inter_pulse_gap_ps=min_gap,
        )
        t0 = time.monotonic()
        result = simulate_one_image_wta(
            resources=resources,
            lasers=lasers,
            class_masks=roi_masks,
            threshold_cond=0.0,
            readout_window_start_ps=readout_start,
            readout_window_end_ps=readout_end,
            readout_stride_steps=calib_stride,
        )
        elapsed = time.monotonic() - t0
        condensed = result["condensed"]
        roi_val = result["class_roi_integrals"][0] if result["class_roi_integrals"] else 0.0
        results.append({"power": p, "condensed": condensed, "roi_integral": roi_val, "elapsed_s": round(elapsed, 2)})
        marker = "COND" if condensed else "sub "
        print(f"    P={p:.0f}: {marker}  roi={roi_val:.3e}  t={elapsed:.1f}s")

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
            print(f"  WARNING: all points condensed — threshold below {results[0]['power']:.0f}")
        elif none_cond:
            print(f"  WARNING: no points condensed — threshold above {results[-1]['power']:.0f}")

    return {
        "sweeps": results,
        "estimated_threshold": threshold,
        "bracket_found": bracket_found,
        "spot_x_um": x_spot,
        "spot_y_um": y_spot,
    }


def _calibrate_threshold_cond(
    resources: SharedSimResources,
    xs_um: np.ndarray,
    ys_um: np.ndarray,
    P_th: float,
    alpha: float,
    enc_cfg: dict,
    readout_cfg: dict,
    n_blank: int = 5,
) -> tuple[float, dict]:
    """Run equal-score (blank) simulations and derive a deterministic reference threshold.

    All n_blank runs use the same fixed seed and identical powers (alpha * P_th for all
    classes).  Because the simulation is deterministic, repeated runs produce the same
    result; n_blank > 1 is retained for future noise studies.  The threshold is set to
    2 × the observed max-ROI so that a clearly condensed class spot (integral >> blank)
    is detected while the blank itself does not trigger.

    Returns
    -------
    threshold_cond : float
    blank_stats    : dict with keys max_roi_values, blank_condensed_count, method
    """
    sigma_space = float(enc_cfg["sigma_space_um"])
    sigma_time = float(enc_cfg["sigma_time_ps"])
    cutoff = float(enc_cfg["cutoff_sigma"])
    power_def = str(enc_cfg["power_definition"])
    pulse_timing_mode = str(enc_cfg.get("pulse_timing_mode", "simultaneous"))
    min_gap = float(enc_cfg.get("min_inter_pulse_gap_ps", 0.0))
    readout_start = float(readout_cfg["window_start_ps"])
    readout_end = float(readout_cfg["window_end_ps"])
    roi_r = float(readout_cfg["roi_radius_um"])
    calib_stride = int(readout_cfg.get("calibration_stride_steps", 500))

    blank_power = alpha * P_th
    powers = np.full(len(xs_um), blank_power)
    grid = resources.grid
    xp = compute_engine.xp

    class_masks = build_class_roi_masks(xs_um, ys_um, roi_r, grid.X, grid.Y, xp)

    all_max_integrals: list[float] = []
    blank_condensed = 0
    for run_i in range(n_blank):
        lasers = build_class_lasers(
            xs_um=xs_um,
            ys_um=ys_um,
            powers=powers,
            sigma_space_um=sigma_space,
            sigma_time_ps=sigma_time,
            cutoff_sigma=cutoff,
            power_definition=power_def,
            grid_X=grid.X,
            grid_Y=grid.Y,
            pulse_timing_mode=pulse_timing_mode,
            min_inter_pulse_gap_ps=min_gap,
        )
        result = simulate_one_image_wta(
            resources=resources,
            lasers=lasers,
            class_masks=class_masks,
            threshold_cond=0.0,
            readout_window_start_ps=readout_start,
            readout_window_end_ps=readout_end,
            readout_stride_steps=calib_stride,
        )
        max_roi = max(result["class_roi_integrals"]) if result["class_roi_integrals"] else 0.0
        all_max_integrals.append(max_roi)
        if result["condensed"]:
            blank_condensed += 1
        print(
            f"  blank {run_i + 1}/{n_blank}: max_roi={max_roi:.3e}  "
            f"condensed={result['condensed']}"
            + (" ← WARNING: blank condensed at alpha*P_th" if result["condensed"] else "")
        )

    if blank_condensed > 0:
        print(
            f"  WARNING: {blank_condensed}/{n_blank} blank runs condensed. "
            "alpha may be too close to threshold. Consider reducing alpha or re-checking P_th."
        )

    median_blank = float(np.median(all_max_integrals))
    threshold_cond = median_blank * 2.0
    print(
        f"  threshold_cond = {threshold_cond:.3e}  "
        f"(deterministic reference: 2 × median blank ROI = {median_blank:.3e})"
    )

    blank_stats = {
        "max_roi_values": all_max_integrals,
        "blank_condensed_count": blank_condensed,
        "n_blank": n_blank,
        "blank_power": float(blank_power),
        "median_blank_roi": median_blank,
        "method": "2x_median_blank_roi_deterministic",
    }
    return threshold_cond, blank_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate WTA single-spot threshold")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    enc_cfg = cfg["encoding"]
    readout_cfg = cfg["readout"]
    calib_cfg = cfg["calibration"]
    geom_cfg = cfg["geometry"]
    amp_cfg = cfg["amplitude"]

    radius = args.radius if args.radius is not None else float(geom_cfg["ring_radius_um"])
    alpha = float(amp_cfg["alpha"])

    p_low = float(calib_cfg["P_single_low"])
    p_high = float(calib_cfg["P_single_high"])
    n_pts = int(calib_cfg["n_sweep_points"])
    n_blank = int(calib_cfg.get("n_blank_samples", 5))
    powers = list(np.linspace(p_low, p_high, n_pts))

    xs_um, ys_um = class_spot_positions(radius)
    x_spot, y_spot = float(xs_um[0]), float(ys_um[0])

    print("=" * 60)
    print(f" mnist_wta_v1 — calibrate_wta  R={radius} μm")
    print("=" * 60)

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    sim_cfg = _build_config(cfg)
    resources = SharedSimResources(sim_cfg)

    print(f"\n=== Single-spot sweep at ({x_spot:.2f}, {y_spot:.2f}) μm ===")
    sweep_result = _run_single_spot_sweep(
        resources, x_spot, y_spot, powers, enc_cfg, readout_cfg
    )

    if not sweep_result["bracket_found"]:
        print(
            "\nERROR: Single-spot threshold not bracketed.\n"
            "Widen P_single_low/P_single_high in config.",
            file=sys.stderr,
        )
        atomic_write_json(os.path.join(run_dir, "calibration_wta.json"), {
            "radius_um": radius,
            "single_spot_sweep": sweep_result,
            "error": "threshold not bracketed",
        })
        sys.exit(1)

    P_th = float(sweep_result["estimated_threshold"])
    print(f"\n  P_th_single_spot = {P_th:.0f}")

    print(f"\n=== Blank runs ({n_blank}× equal score = alpha×P_th) ===")
    threshold_cond, blank_stats = _calibrate_threshold_cond(
        resources, xs_um, ys_um, P_th, alpha, enc_cfg, readout_cfg, n_blank
    )

    result = {
        "radius_um": radius,
        "P_th_single_spot": P_th,
        "threshold_cond": threshold_cond,
        "alpha_for_blank": alpha,
        "n_blank_samples": n_blank,
        "pulse_config": {
            "laser_type": "pulse-gaussian",
            "sigma_time_ps": float(enc_cfg["sigma_time_ps"]),
            "cutoff_sigma": float(enc_cfg["cutoff_sigma"]),
            "power_definition": str(enc_cfg["power_definition"]),
            "pulse_timing_mode": str(enc_cfg.get("pulse_timing_mode", "simultaneous")),
            "min_inter_pulse_gap_ps": float(enc_cfg.get("min_inter_pulse_gap_ps", 0.0)),
        },
        "blank_stats": blank_stats,
        "single_spot_sweep": sweep_result,
    }
    atomic_write_json(os.path.join(run_dir, "calibration_wta.json"), result)
    print(f"\n  P_th_single_spot = {P_th:.0f}")
    print(f"  threshold_cond   = {threshold_cond:.3e}")
    print(f"  Calibration → {run_dir}/calibration_wta.json")


if __name__ == "__main__":
    main()
