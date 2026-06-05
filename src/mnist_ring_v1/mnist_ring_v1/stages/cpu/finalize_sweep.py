"""CPU stage: aggregate trigger-sweep results and write RUN_REPORT.md.

Reads results_summary.json from each sweep point's run directory,
compiles the comparison table, applies the decision criteria from A.4,
and emits RUN_REPORT.md + a summary to stdout.

Invoked as:
    python -m mnist_ring_v1.stages.cpu.finalize_sweep \\
        --sweep-log <path/to/sweep_run_dirs.txt> \\
        --out-dir   <output_directory>

    or with explicit run dirs:
    python -m mnist_ring_v1.stages.cpu.finalize_sweep \\
        --run-dirs 0.3:/path/to/run1 0.4:/path/to/run2 ... \\
        --out-dir <output_directory>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any


class _NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def _atomic_write(path: str, content: str) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _load_point(frac_str: str, run_dir: str) -> dict[str, Any] | None:
    summary_path = os.path.join(run_dir, "results_summary.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(summary_path):
        print(f"  WARNING: no results_summary.json in {run_dir}", file=sys.stderr)
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    manifest: dict = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    ridge_test = summary.get("ridge_classifier", {}).get("accuracy_test")
    nc_test = summary.get("nearest_centroid", {}).get("accuracy_test")
    baseline = summary.get("baseline_pca_accuracy")
    n_total = summary.get("n_total")
    n_test = summary.get("n_test")
    central_frac = summary.get("central_condensed_fraction")
    pretrigger_frac = summary.get("pretrigger_central_fraction")
    gs = summary.get("global_scalars", {})
    t_cond_mean = gs.get("t_cond_mean_ps")
    t_cond_std = gs.get("t_cond_std_ps")
    t_cond_central_mean = gs.get("t_cond_central_mean_ps")

    sigma_trigger = None
    sigma_ring = None
    metadata_path = os.path.join(run_dir, "metadata")
    if os.path.isdir(metadata_path):
        for fname in sorted(os.listdir(metadata_path)):
            if fname.endswith(".json"):
                with open(os.path.join(metadata_path, fname)) as f:
                    bm = json.load(f)
                rc = bm.get("readout_config", {})
                sigma_trigger = rc.get("sigma_time_trigger_ps")
                sigma_ring = rc.get("sigma_time_ring_ps")
                break

    return {
        "trigger_fraction": float(frac_str),
        "run_dir": run_dir,
        "ridge_test": ridge_test,
        "nc_test": nc_test,
        "baseline_pca": baseline,
        "n_total": n_total,
        "n_test": n_test,
        "central_condensed_fraction": central_frac,
        "pretrigger_central_fraction": pretrigger_frac,
        "t_cond_mean_ps": t_cond_mean,
        "t_cond_std_ps": t_cond_std,
        "t_cond_central_mean_ps": t_cond_central_mean,
        "sigma_time_trigger_ps": sigma_trigger,
        "sigma_time_ring_ps": sigma_ring,
    }


def _decision_verdict(best_ridge_test: float | None) -> str:
    if best_ridge_test is None:
        return "UNKNOWN — no test accuracy available"
    if best_ridge_test > 0.50:
        return f"CONTINUE ring_v1 (best Ridge test={best_ridge_test:.1%} > 50%) — proceed to T_max sweep"
    if best_ridge_test >= 0.40:
        return f"FLAG for owner decision (best Ridge test={best_ridge_test:.1%} is 40-50%)"
    return f"CLOSE ring_v1 (best Ridge test={best_ridge_test:.1%} < 40%) — insufficient improvement"


def _format_val(v: Any, fmt: str = ".3f") -> str:
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _make_report(
    points: list[dict[str, Any]],
    out_dir: str,
    run_date: str,
) -> str:
    points_sorted = sorted(points, key=lambda p: p["trigger_fraction"])

    best_ridge = max(
        (p["ridge_test"] for p in points_sorted if p["ridge_test"] is not None),
        default=None,
    )
    verdict = _decision_verdict(best_ridge)

    baseline_ref = None
    for p in points_sorted:
        if abs(p["trigger_fraction"] - 0.8) < 1e-6 and p["baseline_pca"] is not None:
            baseline_ref = p["baseline_pca"]
            break

    col_w = [18, 12, 12, 12, 10, 10, 12, 12, 14]
    headers = [
        "trig_frac", "ridge_test", "nc_test", "baseline",
        "cond_frac", "pretrig", "t_cond_mean", "t_cond_std", "delta_vs_base",
    ]

    def row_str(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"

    table_rows = [row_str(headers), sep]
    for p in points_sorted:
        rt = p.get("ridge_test")
        base = p.get("baseline_pca")
        delta = (rt - base) if (rt is not None and base is not None) else None
        cells = [
            f"{p['trigger_fraction']:.2f}",
            _format_val(rt, ".3f"),
            _format_val(p.get("nc_test"), ".3f"),
            _format_val(base, ".3f"),
            _format_val(p.get("central_condensed_fraction"), ".3f"),
            _format_val(p.get("pretrigger_central_fraction"), ".3f"),
            _format_val(p.get("t_cond_mean_ps"), ".1f"),
            _format_val(p.get("t_cond_std_ps"), ".1f"),
            _format_val(delta, "+.3f") if delta is not None else "N/A",
        ]
        table_rows.append(row_str(cells))

    table = "\n".join(table_rows)

    baseline_check = ""
    p08 = next((p for p in points_sorted if abs(p["trigger_fraction"] - 0.8) < 1e-6), None)
    if p08 is not None:
        rt08 = p08.get("ridge_test")
        ref_acc = 0.31
        if rt08 is not None:
            diff = abs(rt08 - ref_acc)
            status = "PASS" if diff <= 0.05 else "FAIL — check sigma_time split"
            baseline_check = (
                f"\n### A.6 Baseline replication check (trigger_fraction=0.8)\n\n"
                f"Previous reference: {ref_acc:.1%}  |  "
                f"Current: {rt08:.1%}  |  "
                f"Diff: {diff:+.1%}  |  "
                f"Status: **{status}** (tolerance ±5pp)\n"
            )
        else:
            baseline_check = (
                "\n### A.6 Baseline replication check\n\n"
                "trigger_fraction=0.8 point has no test accuracy — run not complete.\n"
            )

    risks = """
### Physical risks observed

1. **WTA degeneration to argmax threshold**: If `central_condensed_fraction` is either 0% (no condensation) or 100% (always condenses regardless of digit), the ring+trigger system is acting as a threshold gate, not encoding digit-class information. Check per-digit spread in `t_cond` and `psi_sq_max`.

2. **Pre-trigger central condensation**: If `pretrig > 0`, the ring power is too high and the trigger is not providing the decisive gate. Reduce `P_ring_subthreshold_fraction` and rerun calibration.

3. **sigma_time_trigger_ps=0.1 effect**: A 10× sharper trigger pulse changes the calibrated trigger threshold. If baseline replication (A.6) fails, the sigma_time split in calibrate.py was not correctly applied to the trigger calibration sweep.
"""

    runs_table_rows = ["| trigger_fraction | run_dir |", "|-|-|"]
    for p in points_sorted:
        runs_table_rows.append(f"| {p['trigger_fraction']:.2f} | `{p['run_dir']}` |")
    runs_table = "\n".join(runs_table_rows)

    report = f"""# mnist_ring_v1 Trigger Sweep Report

**Date**: {run_date}
**Output dir**: `{out_dir}`
**sigma_time_ring_ps**: 1.5 ps  |  **sigma_time_trigger_ps**: 0.1 ps
**Images per point**: ~200 (10 train + 10 test per class)

## Results

{table}

## Decision (A.4)

**Verdict**: {verdict}

| Criterion | Threshold | Outcome |
|-----------|-----------|---------|
| best Ridge test > 50% | continue ring_v1 | {"YES" if best_ridge is not None and best_ridge > 0.50 else "NO"} |
| best Ridge test 40–50% | flag for owner | {"YES" if best_ridge is not None and 0.40 <= best_ridge <= 0.50 else "NO"} |
| best Ridge test < 40% | close ring_v1 | {"YES" if best_ridge is not None and best_ridge < 0.40 else "NO"} |
{baseline_check}
## Setup

| Parameter | Value |
|-----------|-------|
| sigma_time_ring_ps | 1.5 |
| sigma_time_trigger_ps | 0.1 |
| ring sigma_time fixed? | yes |
| sweep values | 0.3, 0.4, 0.5, 0.6, 0.8 |
| images per point | 200 (balanced 20/class) |
| PCA components | 16 |
| T_max | 100 ps |

## Run directories

{runs_table}
{risks}
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize trigger sweep results")
    parser.add_argument(
        "--sweep-log",
        default=None,
        help="Text file with lines 'trigger_fraction -> run_dir'",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help="Explicit run dirs as 'frac:path' pairs, e.g. 0.3:/path/to/run",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if args.sweep_log is None and args.run_dirs is None:
        print("ERROR: provide --sweep-log or --run-dirs", file=sys.stderr)
        sys.exit(1)

    frac_run_pairs: list[tuple[str, str]] = []
    if args.sweep_log:
        with open(args.sweep_log) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("->")
                if len(parts) == 2:
                    frac_run_pairs.append((parts[0].strip(), parts[1].strip()))

    if args.run_dirs:
        for spec in args.run_dirs:
            frac_str, _, run_dir = spec.partition(":")
            if run_dir:
                frac_run_pairs.append((frac_str.strip(), run_dir.strip()))

    if not frac_run_pairs:
        print("ERROR: no sweep points found", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" mnist_ring_v1 — finalize_sweep")
    print("=" * 60)
    print(f"  {len(frac_run_pairs)} sweep points")

    points: list[dict[str, Any]] = []
    for frac_str, run_dir in frac_run_pairs:
        print(f"  trig_frac={frac_str}  run_dir={run_dir}")
        point = _load_point(frac_str, run_dir)
        if point is not None:
            points.append(point)

    if not points:
        print("ERROR: no valid sweep points loaded", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_md = _make_report(points, args.out_dir, run_date)
    report_path = os.path.join(args.out_dir, "RUN_REPORT.md")
    _atomic_write(report_path, report_md)

    best_ridge = max(
        (p["ridge_test"] for p in points if p["ridge_test"] is not None),
        default=None,
    )
    verdict = _decision_verdict(best_ridge)

    sweep_summary_path = os.path.join(args.out_dir, "sweep_summary.json")
    fd, tmp = tempfile.mkstemp(dir=args.out_dir, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "n_points": len(points),
                    "best_ridge_test": best_ridge,
                    "verdict": verdict,
                    "points": points,
                    "run_date": run_date,
                },
                f,
                indent=2,
                cls=_NpEncoder,
            )
            f.write("\n")
    except Exception:
        os.unlink(tmp)
        raise
    os.replace(tmp, sweep_summary_path)

    print(f"\n  Report   : {report_path}")
    print(f"  Summary  : {sweep_summary_path}")
    print(f"\n  VERDICT  : {verdict}")

    print("\n--- Sweep table (stdout) ---")
    col_w = [10, 10, 10, 12]
    hdr = ["trig_frac", "ridge", "nc", "baseline"]
    sep = "-" * (sum(col_w) + 3 * len(col_w))
    print(sep)
    print("  ".join(h.ljust(w) for h, w in zip(hdr, col_w)))
    print(sep)
    for p in sorted(points, key=lambda x: x["trigger_fraction"]):
        print("  ".join(
            _format_val(p.get(k), ".3f").ljust(w)
            for k, w in zip(
                ["trigger_fraction", "ridge_test", "nc_test", "baseline_pca"],
                col_w,
            )
        ))
    print(sep)


if __name__ == "__main__":
    main()
