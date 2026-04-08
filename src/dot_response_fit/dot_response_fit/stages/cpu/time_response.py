"""CPU Stage 1: generate the target amplitude-response curve via the 1-D ODE.

Reads the ``time_response`` and ``global.physics`` sections of config.yaml,
runs :func:`polarism.response.sweep.run_amplitude_sweep`, and writes
``target_response.json`` to the run directory.  Also saves a summary PNG.

Invoked as::

    python -m dot_response_fit.stages.cpu.time_response \\
        --config <run_dir>/config.yaml --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

from dot_response_fit.config.loader import (
    build_amplitude_array,
    build_t_eval,
    extract_time_response_cfg,
    load_config,
    make_dataclass,
)
from dot_response_fit.manifest.io import (
    atomic_write_json,
    set_manifest_field,
)
from polarism.config.simulation_parameters import PhysicsConstants
from polarism.response.time_only import params_from_physics
from polarism.response.sweep import run_amplitude_sweep


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU Stage 1: generate target time response")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    global_cfg = cfg["global"]
    tr_cfg = extract_time_response_cfg(cfg)

    physics = make_dataclass(PhysicsConstants, global_cfg.get("physics", {}))
    params = params_from_physics(physics)

    t_eval = build_t_eval(tr_cfg)
    amplitudes = build_amplitude_array(tr_cfg["amplitudes"])
    pulse_cfg = tr_cfg["pulse"]
    center = float(pulse_cfg["center"])
    width_fwhm = float(pulse_cfg["width_fwhm"])

    print("=" * 60)
    print(" CPU Stage 1: Target Time Response")
    print("=" * 60)
    print(f"  Run dir     : {run_dir}")
    print(f"  Amplitudes  : {len(amplitudes)} values  [{amplitudes[0]:.1f}, {amplitudes[-1]:.1f}]")
    print(f"  t_eval      : [{t_eval[0]:.1f}, {t_eval[-1]:.1f}] ps  ({len(t_eval)} points)")
    print(f"  Pulse       : center={center} ps, FWHM={width_fwhm} ps")
    print(f"  Params      : {params}")
    print()

    result = run_amplitude_sweep(amplitudes, center, width_fwhm, t_eval, params)

    out_path = os.path.join(run_dir, "target_response.json")
    atomic_write_json(out_path, result)
    print(f"  Saved: {out_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    _save_target_plot(result, results_dir)

    try:
        set_manifest_field(run_dir, "target_response_complete", True)
        set_manifest_field(run_dir, "target_response_path", out_path)
    except Exception as exc:
        print(f"  WARNING: could not update manifest: {exc}", file=sys.stderr)

    print("\n  Stage 1 complete.")


def _save_target_plot(result: dict, results_dir: str) -> None:
    """Save a summary PNG of the amplitude-response curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        amplitudes = result["amplitudes"]
        nc_integrals = result["nc_integrals"]

        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        ax.plot(amplitudes, nc_integrals, "b-", linewidth=1.5)
        ax.set_xlabel("Pump amplitude")
        ax.set_ylabel(r"$\int n_C(t)\,dt$")
        ax.set_title("Target: integral of $n_C(t)$ vs pump amplitude")
        ax.grid(True, alpha=0.3)

        out_path = os.path.join(results_dir, "target_response.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  Plot: {out_path}")
    except Exception as exc:
        print(f"  WARNING: could not save target plot: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
