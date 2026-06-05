"""Compare two completed mnist_wta_v1 pilot runs.

Invoked as:
    python -m mnist_wta_v1.stages.cpu.compare_pilots \\
        --run-a <R8 run_dir> --run-b <R15 run_dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any

import numpy as np

from mnist_common.io.atomic import atomic_write_json


def _atomic_write_text(path: str, text: str) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_scalars(run_dir: str) -> list[dict[str, Any]]:
    meta_dir = os.path.join(run_dir, "metadata")
    if not os.path.isdir(meta_dir):
        return []
    scalars: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(meta_dir)):
        if not (fname.startswith("batch_") and fname.endswith(".json")):
            continue
        batch = _load_json(os.path.join(meta_dir, fname))
        scalars.extend(batch.get("scalars", []))
    return scalars


def _margin_stats(scalars: list[dict[str, Any]]) -> dict[str, float | None]:
    margins = np.array(
        [float(s["margin"]) for s in scalars if s.get("margin") is not None],
        dtype=np.float64,
    )
    if len(margins) == 0:
        return {"p10": None, "p50": None, "p90": None, "mean": None}
    return {
        "p10": float(np.percentile(margins, 10)),
        "p50": float(np.percentile(margins, 50)),
        "p90": float(np.percentile(margins, 90)),
        "mean": float(np.mean(margins)),
    }


def _run_record(run_dir: str, label: str) -> dict[str, Any]:
    summary_path = os.path.join(run_dir, "results_summary_wta.json")
    manifest_path = os.path.join(run_dir, "manifest.json")
    calib_path = os.path.join(run_dir, "calibration_wta.json")
    if not os.path.isfile(summary_path):
        print(f"ERROR: missing results_summary_wta.json: {summary_path}", file=sys.stderr)
        sys.exit(1)

    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path) if os.path.isfile(manifest_path) else {}
    calib = _load_json(calib_path) if os.path.isfile(calib_path) else {}
    scalars_all = _load_scalars(run_dir)
    test_scalars = [s for s in scalars_all if s.get("split") == "test"]

    n_test = len(test_scalars)
    n_global_condensed = sum(1 for s in test_scalars if s.get("condensed"))
    n_roi_condensed = sum(1 for s in test_scalars if not s.get("no_cond", True))
    n_multi = sum(1 for s in test_scalars if s.get("multi_cond"))

    return {
        "label": label,
        "run_dir": run_dir,
        "radius_um": calib.get("radius_um", manifest.get("ring_radius_um")),
        "accuracy_test": summary.get("accuracy_test"),
        "baseline_ridge_accuracy": summary.get("baseline_ridge_accuracy"),
        "delta_vs_baseline": summary.get("delta_vs_baseline"),
        "accuracy_test_excl_no_cond": summary.get("accuracy_test_excl_no_cond"),
        "accuracy_test_margin_gt_2_excl_no_cond": summary.get("accuracy_test_margin_gt_2_excl_no_cond"),
        "accuracy_test_margin_gt_5_excl_no_cond": summary.get("accuracy_test_margin_gt_5_excl_no_cond"),
        "n_test": summary.get("n_test", n_test),
        "global_condensed_fraction_test": n_global_condensed / max(n_test, 1),
        "roi_condensed_fraction_test": n_roi_condensed / max(n_test, 1),
        "no_cond_rate_test": summary.get("no_cond_rate_test"),
        "multi_cond_rate_test": n_multi / max(n_test, 1),
        "margin_stats_test": _margin_stats(test_scalars),
        "P_th_single_spot": calib.get("P_th_single_spot"),
        "threshold_cond": calib.get("threshold_cond"),
        "blank_condensed_count": calib.get("blank_stats", {}).get("blank_condensed_count"),
    }


def _fmt(v: Any, precision: int = 3) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{precision}f}"
    except (TypeError, ValueError):
        return str(v)


def _make_report(a: dict[str, Any], b: dict[str, Any], out_dir: str) -> str:
    delta_a = a.get("delta_vs_baseline")
    delta_b = b.get("delta_vs_baseline")
    delta_a_minus_b = (
        float(delta_a) - float(delta_b)
        if delta_a is not None and delta_b is not None
        else None
    )

    headers = [
        "pilot", "R_um", "acc", "baseline", "delta",
        "acc_cond", "cond_roi", "multi", "margin_p50", "margin_p90",
    ]
    widths = [8, 8, 8, 10, 9, 10, 10, 8, 11, 11]

    def row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    rows = [row(headers), sep]
    for rec in (a, b):
        margins = rec["margin_stats_test"]
        rows.append(row([
            str(rec["label"]),
            _fmt(rec.get("radius_um"), 1),
            _fmt(rec.get("accuracy_test")),
            _fmt(rec.get("baseline_ridge_accuracy")),
            _fmt(rec.get("delta_vs_baseline"), 4),
            _fmt(rec.get("accuracy_test_excl_no_cond")),
            _fmt(rec.get("roi_condensed_fraction_test")),
            _fmt(rec.get("multi_cond_rate_test")),
            _fmt(margins.get("p50")),
            _fmt(margins.get("p90")),
        ]))

    table = "\n".join(rows)
    return f"""# mnist_wta_v1 Pilot A/B Report

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Output dir**: `{out_dir}`

## Comparison

{table}

## Delta Contrast

`delta(A) - delta(B) = {_fmt(delta_a_minus_b, 4)}`

Interpretacja:

- A powinno reprezentowac silniejsze sprzezenie przestrzenne.
- B jest kontrola weak-coupling. Jesli B dziala podobnie jak A, wynik jest
  podejrzany jako zwykly argmax/prog lokalny, nie polariton-mediated WTA.
- Do argumentu fizycznego uzywaj przede wszystkim `acc_cond`, `cond_roi`,
  `multi` oraz percentyli marginu, nie samego `accuracy_test`.

## Run Directories

| pilot | run_dir |
|-|-|
| {a['label']} | `{a['run_dir']}` |
| {b['label']} | `{b['run_dir']}` |
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare mnist_wta_v1 pilot A/B runs")
    parser.add_argument("--run-a", required=True, help="Pilot A run_dir, usually R=8")
    parser.add_argument("--run-b", required=True, help="Pilot B run_dir, usually R=15")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(run_a)
    os.makedirs(out_dir, exist_ok=True)

    rec_a = _run_record(run_a, args.label_a)
    rec_b = _run_record(run_b, args.label_b)
    delta_a = rec_a.get("delta_vs_baseline")
    delta_b = rec_b.get("delta_vs_baseline")
    delta_a_minus_b = (
        float(delta_a) - float(delta_b)
        if delta_a is not None and delta_b is not None
        else None
    )

    summary = {
        "pilot_a": rec_a,
        "pilot_b": rec_b,
        "delta_a_minus_delta_b": delta_a_minus_b,
    }
    atomic_write_json(os.path.join(out_dir, "pilot_ab_summary.json"), summary)

    report = _make_report(rec_a, rec_b, out_dir)
    report_path = os.path.join(out_dir, "PILOT_AB_REPORT.md")
    _atomic_write_text(report_path, report)

    print("=" * 60)
    print(" mnist_wta_v1 — compare_pilots")
    print("=" * 60)
    print(f"  A delta: {_fmt(delta_a, 4)}")
    print(f"  B delta: {_fmt(delta_b, 4)}")
    print(f"  delta(A)-delta(B): {_fmt(delta_a_minus_b, 4)}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
