"""CPU stage: cross-scenario finalize.

Aggregates per-scenario metadata into ``results_summary.json`` and generates
the cross-scenario comparison plot from scalar sidecars.  Does not require
per-scenario HDF5 files — reads only metadata JSON and ``*_scalars.npz``.

Invoked by Slurm as:
    python -m pipeline.stages.cpu.finalize --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

from pipeline.manifest.io import (
    atomic_write_json,
    load_scenario_index,
    load_scenario_meta,
    scenario_meta_path,
    set_manifest_field,
)
from pipeline.stages.cpu.viz_engine import (
    generate_summary,
    generate_sweep_diagnostics,
    generate_sweep_heatmaps,
)


_PROBE5_CSV_COLUMNS = [
    "scenario", "probe_delay_ps", "probe_energy", "ring_energy", "ring_scale",
    "square_side_um", "sigma_space", "pump_end_ps",
    "central_roi_peak_psi_sq", "central_roi_peak_emission",
    "central_roi_peak_nR", "t_peak_emission_ps",
    "crossed_threshold",
]

_PROBE5_THRESHOLD_CRITERION = 0.05


_FRINGE_CSV_COLUMNS = [
    "scenario", "square_side_um", "energy", "sigma_space",
    "fringe_contrast_max", "t_fringe_contrast_max_ps",
    "fringe_spacing_at_max_contrast_um", "fringe_fft_peak_k_at_max_contrast",
    "fringe_cv_max", "h_contrast_max", "v_contrast_max",
    "fringe_window_psi_sq_max", "t_fringe_window_psi_sq_max_ps",
    "central_roi_peak_psi_sq", "central_roi_peak_emission",
    "crossed_threshold",
]


def _load_fringe_json(run_dir: str, scenario_name: str) -> dict | None:
    path = os.path.join(run_dir, f"{scenario_name}_fringe.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _probe5_threshold_criterion(run_dir: str) -> float:
    """Return the strict psi² criterion used for probe5 condensation labels."""
    path = os.path.join(run_dir, "threshold_result.json")
    if not os.path.isfile(path):
        return _PROBE5_THRESHOLD_CRITERION
    try:
        with open(path) as f:
            threshold = json.load(f)
        return float(
            (threshold.get("calibration") or {}).get(
                "condensation_psi_sq_threshold",
                _PROBE5_THRESHOLD_CRITERION,
            )
        )
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        return _PROBE5_THRESHOLD_CRITERION


def _load_sidecar_roi_peaks(
    run_dir: str, scenario_name: str
) -> dict[str, float]:
    path = os.path.join(run_dir, f"{scenario_name}_scalars.npz")
    if not os.path.isfile(path):
        return {}
    data = np.load(path)
    out: dict[str, float] = {}
    prefix = "roi_center_D_circle"
    for suffix, key in (
        ("_mean_psi_sq", "central_roi_peak_psi_sq"),
        ("_integral_emission", "central_roi_peak_emission"),
    ):
        arr_key = f"{prefix}{suffix}"
        if arr_key in data:
            out[key] = float(data[arr_key].max())
    return out


def _generate_fringe_summary(
    scenarios: list[str], run_dir: str, results_dir: str
) -> None:
    rows: list[dict] = []
    for name in scenarios:
        fringe = _load_fringe_json(run_dir, name)
        if fringe is None or "aggregated" not in fringe:
            continue
        meta = load_scenario_meta(run_dir, name)
        sweep = meta.get("sweep") or {}
        agg = fringe["aggregated"]
        roi = _load_sidecar_roi_peaks(run_dir, name)
        row: dict = {
            "scenario": name,
            "square_side_um": sweep.get("square_side", float("nan")),
            "energy": sweep.get("power", float("nan")),
            "sigma_space": sweep.get("sigma_space", float("nan")),
            "fringe_contrast_max": agg.get("fringe_contrast_max", float("nan")),
            "t_fringe_contrast_max_ps": agg.get("t_fringe_contrast_max", float("nan")),
            "fringe_spacing_at_max_contrast_um": agg.get("fringe_spacing_at_max_contrast", float("nan")),
            "fringe_fft_peak_k_at_max_contrast": agg.get("fringe_fft_peak_k_at_max_contrast", float("nan")),
            "fringe_cv_max": agg.get("fringe_cv_max", float("nan")),
            "h_contrast_max": agg.get("h_contrast_max", float("nan")),
            "v_contrast_max": agg.get("v_contrast_max", float("nan")),
            "fringe_window_psi_sq_max": agg.get("fringe_window_psi_sq_max", float("nan")),
            "t_fringe_window_psi_sq_max_ps": agg.get("t_fringe_window_psi_sq_max", float("nan")),
            "central_roi_peak_psi_sq": roi.get("central_roi_peak_psi_sq", float("nan")),
            "central_roi_peak_emission": roi.get("central_roi_peak_emission", float("nan")),
            "crossed_threshold": agg.get("crossed_threshold", False),
        }
        rows.append(row)

    if not rows:
        return

    csv_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FRINGE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "spatiotemporal_square4_fringe_summary.json")
    atomic_write_json(json_path, rows)

    _generate_selected_extremes(rows, results_dir)
    print(f"  Fringe summary: {csv_path}  ({len(rows)} scenarios)")


def _generate_selected_extremes(rows: list[dict], results_dir: str) -> None:
    if not rows:
        return

    def _pick(key: str, best_fn) -> dict | None:
        valid = [r for r in rows if not _is_nan(r.get(key))]
        return best_fn(valid, key=lambda r: r[key]) if valid else None

    def _is_nan(v) -> bool:
        try:
            return v is None or float(v) != float(v)
        except (TypeError, ValueError):
            return True

    above = [r for r in rows if r.get("crossed_threshold")]
    below = [r for r in rows if not r.get("crossed_threshold")]

    extremes: dict = {}
    r = _pick("central_roi_peak_psi_sq", max)
    if r:
        extremes["max_central_psi_sq"] = r
    r = _pick("central_roi_peak_psi_sq", min)
    if r:
        extremes["min_central_psi_sq"] = r
    r = _pick("fringe_contrast_max", max)
    if r:
        extremes["max_fringe_contrast"] = r
    if above:
        extremes["first_above_threshold"] = min(
            above, key=lambda r: r.get("energy", float("inf"))
        )
    if below:
        valid_below = [r for r in below if not _is_nan(r.get("central_roi_peak_psi_sq"))]
        if valid_below:
            extremes["closest_below_threshold"] = max(
                valid_below, key=lambda r: r["central_roi_peak_psi_sq"]
            )

    atomic_write_json(os.path.join(results_dir, "selected_extremes.json"), extremes)


def _post_probe_max(
    data: "np.ndarray",
    time_arr: "np.ndarray | None",
    key: str,
    probe_delay: float,
) -> float:
    if key not in data:
        return float("nan")
    arr = data[key]
    if time_arr is not None and len(time_arr) == len(arr):
        arr = arr[time_arr >= probe_delay]
    return float(arr.max()) if len(arr) > 0 else float("nan")


def _time_of_max(
    data: "np.ndarray",
    time_arr: "np.ndarray | None",
    key: str,
    probe_delay: float,
) -> float:
    if key not in data or time_arr is None:
        return float("nan")
    arr = data[key]
    if len(time_arr) != len(arr):
        return float("nan")
    mask = time_arr >= probe_delay
    arr_post = arr[mask]
    t_post = time_arr[mask]
    if len(arr_post) == 0:
        return float("nan")
    return float(t_post[arr_post.argmax()])


def _generate_probe5_heatmap(
    rows: list[dict],
    results_dir: str,
    metric: str,
    filename: str,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    delays = sorted({float(r["probe_delay_ps"]) for r in rows if r.get("probe_delay_ps") is not None})
    energies = sorted({float(r["probe_energy"]) for r in rows if r.get("probe_energy") is not None})
    if not delays or not energies:
        return

    z = np.full((len(energies), len(delays)), float("nan"))
    delay_idx = {d: i for i, d in enumerate(delays)}
    energy_idx = {e: i for i, e in enumerate(energies)}

    for r in rows:
        d = r.get("probe_delay_ps")
        e = r.get("probe_energy")
        v = r.get(metric)
        if d is None or e is None or v is None:
            continue
        if float(d) in delay_idx and float(e) in energy_idx and np.isfinite(float(v)):
            z[energy_idx[float(e)], delay_idx[float(d)]] = float(v)

    fig, ax = plt.subplots(figsize=(max(6, len(delays) * 0.55), max(4, len(energies) * 0.28)))
    im = ax.pcolormesh(delays, energies, z, shading="nearest", cmap="hot")
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel("probe delay (ps)")
    ax.set_ylabel("probe energy")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close(fig)


def _unique_float_values(rows: list[dict], key: str) -> list[float]:
    values: set[float] = set()
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fvalue):
            values.add(fvalue)
    return sorted(values)


def _compact_float_label(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _generate_probe5_ring_heatmap(
    rows: list[dict],
    results_dir: str,
    metric: str,
    filename: str,
    title: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x_key = "ring_scale"
    x_values = _unique_float_values(rows, x_key)
    x_label = "ring scale"
    if not x_values:
        x_key = "ring_energy"
        x_values = _unique_float_values(rows, x_key)
        x_label = "ring energy"

    probe_energies = _unique_float_values(rows, "probe_energy")
    if not x_values or not probe_energies:
        return

    z = np.full((len(probe_energies), len(x_values)), float("nan"))
    x_idx = {x: i for i, x in enumerate(x_values)}
    e_idx = {e: i for i, e in enumerate(probe_energies)}

    for row in rows:
        x = row.get(x_key)
        e = row.get("probe_energy")
        v = row.get(metric)
        if x is None or e is None or v is None:
            continue
        try:
            xf = float(x)
            ef = float(e)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if xf in x_idx and ef in e_idx and np.isfinite(vf):
            z[e_idx[ef], x_idx[xf]] = vf

    fig, ax = plt.subplots(
        figsize=(max(6, len(x_values) * 0.55), max(4, len(probe_energies) * 0.45))
    )
    im = ax.pcolormesh(x_values, probe_energies, z, shading="nearest", cmap="hot")
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel(x_label)
    ax.set_ylabel("probe energy")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.close(fig)


def _generate_probe5_selected_cases(rows: list[dict], results_dir: str) -> None:
    above = [r for r in rows if r.get("crossed_threshold")]
    below = [
        r for r in rows
        if not r.get("crossed_threshold")
        and np.isfinite(float(r.get("central_roi_peak_psi_sq", float("nan"))))
    ]

    cases: dict = {}
    if rows:
        cases["max_response"] = max(
            rows, key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0)
        )
        cases["min_response"] = min(
            rows, key=lambda r: float(r.get("central_roi_peak_psi_sq") or float("inf"))
        )
    if above:
        cases["first_above_threshold"] = min(
            above, key=lambda r: float(r.get("probe_energy") or float("inf"))
        )
    if below:
        cases["closest_below_threshold"] = max(
            below, key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0)
        )

    probe_rows = [r for r in rows if float(r.get("probe_energy") or 0.0) > 0.0]
    ring_only_rows = [r for r in rows if float(r.get("probe_energy") or 0.0) == 0.0]
    above_probe = [r for r in probe_rows if r.get("crossed_threshold")]
    below_probe = [
        r for r in probe_rows
        if not r.get("crossed_threshold")
        and np.isfinite(float(r.get("central_roi_peak_psi_sq", float("nan"))))
    ]
    if above_probe:
        cases["lowest_ring_above_threshold"] = min(
            above_probe, key=lambda r: float(r.get("ring_energy") or float("inf"))
        )
    if below_probe:
        cases["highest_ring_below_threshold"] = max(
            below_probe, key=lambda r: float(r.get("ring_energy") or 0.0)
        )
    if ring_only_rows:
        cases["ring_only_max_response"] = max(
            ring_only_rows,
            key=lambda r: float(r.get("central_roi_peak_psi_sq") or 0.0),
        )

    atomic_write_json(
        os.path.join(results_dir, "selected_probe5_threshold_cases.json"), cases
    )


def _generate_probe5_summary(
    scenarios: list[str], run_dir: str, results_dir: str
) -> None:
    threshold_criterion = _probe5_threshold_criterion(run_dir)
    rows: list[dict] = []
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        sweep = meta.get("sweep") or {}
        if sweep.get("mode") != "probe5":
            continue

        sidecar_path = os.path.join(run_dir, f"{name}_scalars.npz")
        if not os.path.isfile(sidecar_path):
            continue

        data = np.load(sidecar_path)
        probe_delay = float(sweep.get("probe_delay_ps", 0.0))
        time_arr = data.get("time", None)
        prefix = "roi_center_D_circle"

        row: dict = {
            "scenario": name,
            "probe_delay_ps": probe_delay,
            "probe_energy": float(sweep.get("probe_energy", float("nan"))),
            "ring_energy": float(sweep.get("ring_energy", float("nan"))),
            "ring_scale": float(sweep.get("ring_scale", float("nan"))),
            "square_side_um": float(sweep.get("square_side", float("nan"))),
            "sigma_space": float(sweep.get("sigma_space", float("nan"))),
            "pump_end_ps": float(sweep.get("pump_end_ps", float("nan"))),
            "central_roi_peak_psi_sq": _post_probe_max(
                data, time_arr, f"{prefix}_mean_psi_sq", probe_delay
            ),
            "central_roi_peak_emission": _post_probe_max(
                data, time_arr, f"{prefix}_integral_emission", probe_delay
            ),
            "central_roi_peak_nR": _post_probe_max(
                data, time_arr, f"{prefix}_mean_nR", probe_delay
            ),
            "t_peak_emission_ps": _time_of_max(
                data, time_arr, f"{prefix}_integral_emission", probe_delay
            ),
        }
        row["crossed_threshold"] = (
            np.isfinite(row["central_roi_peak_psi_sq"])
            and row["central_roi_peak_psi_sq"] > threshold_criterion
        )
        rows.append(row)

    if not rows:
        return

    csv_path = os.path.join(results_dir, "probe5_threshold_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PROBE5_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(results_dir, "probe5_threshold_summary.json")
    atomic_write_json(json_path, rows)

    try:
        ring_values = _unique_float_values(rows, "ring_energy")
        delay_values = _unique_float_values(rows, "probe_delay_ps")
        if len(ring_values) > 1:
            for delay in delay_values:
                delay_rows = [
                    r for r in rows
                    if abs(float(r.get("probe_delay_ps", float("nan"))) - delay) < 1e-9
                ]
                suffix = (
                    f"_d{_compact_float_label(delay)}"
                    if len(delay_values) > 1
                    else ""
                )
                _generate_probe5_ring_heatmap(
                    delay_rows,
                    results_dir,
                    "central_roi_peak_psi_sq",
                    f"probe5_ring_sweep_heatmap_psi_sq{suffix}.png",
                    (
                        "Probe5 central |ψ|² peak vs ring strength"
                        f"  (threshold={threshold_criterion})"
                    ),
                )
                _generate_probe5_ring_heatmap(
                    delay_rows,
                    results_dir,
                    "central_roi_peak_emission",
                    f"probe5_ring_sweep_heatmap_emission{suffix}.png",
                    "Probe5 central emission peak vs ring strength",
                )
        else:
            _generate_probe5_heatmap(
                rows,
                results_dir,
                "central_roi_peak_psi_sq",
                "probe5_threshold_heatmap_psi_sq.png",
                f"Probe5 central |ψ|² peak  (threshold={threshold_criterion})",
            )
            _generate_probe5_heatmap(
                rows,
                results_dir,
                "central_roi_peak_emission",
                "probe5_threshold_heatmap_emission.png",
                "Probe5 central emission peak",
            )
    except Exception as e:
        print(f"  WARNING: probe5 heatmap generation failed: {e}", file=sys.stderr)

    try:
        _generate_probe5_selected_cases(rows, results_dir)
    except Exception as e:
        print(f"  WARNING: probe5 selected cases failed: {e}", file=sys.stderr)

    print(f"  Probe5 summary: {csv_path}  ({len(rows)} scenarios)")


def _check_artifacts(run_dir: str, scenarios: list[str]) -> list[str]:
    """Return error strings for any missing required per-scenario artifact."""
    errors: list[str] = []
    for name in scenarios:
        meta = scenario_meta_path(run_dir, name)
        sidecar = os.path.join(run_dir, f"{name}_scalars.npz")
        if not os.path.isfile(meta):
            errors.append(f"Missing metadata for scenario '{name}': {meta}")
        if not os.path.isfile(sidecar):
            errors.append(f"Missing scalar sidecar for scenario '{name}': {sidecar}")
    return errors


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU finalize / cross-scenario summary")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    try:
        scenarios = load_scenario_index(run_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(" CPU Finalize")
    print("=" * 60)
    print(f"  Run dir   : {run_dir}")
    print(f"  Scenarios : {scenarios}")

    errors = _check_artifacts(run_dir, scenarios)
    if errors:
        print("\nFINALIZE ABORTED — missing artifacts:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    import json
    threshold_path = os.path.join(run_dir, "threshold_result.json")
    with open(threshold_path) as f:
        threshold = json.load(f)

    is_sweep = threshold.get("mode") == "parameter_sweep"
    routines_summary: dict = {}
    sweep_metas: list[dict] = []
    effective_powers: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        t_cond = meta.get("t_cond")
        routines_summary[name] = {
            "t_cond": t_cond,
            "n_lasers": meta.get("n_lasers"),
            "phase_offsets": meta.get("phase_offsets"),
        }
        if meta.get("sweep"):
            sweep_metas.append(meta)
        if is_sweep and meta.get("effective_power") is not None:
            effective_powers[name] = meta["effective_power"]
        print(
            f"  {name}: t_cond="
            f"{'%.1f ps' % t_cond if t_cond is not None else 'NO CONDENSATION'}"
        )

    power_definitions: dict = {}
    for name in scenarios:
        meta = load_scenario_meta(run_dir, name)
        pd = meta.get("effective_power_definition")
        if pd:
            power_definitions[name] = pd

    summary = {
        "run_dir": run_dir,
        "mode": threshold.get("mode", "threshold_search"),
        "power_definition": threshold.get("power_definition", "peak_amplitude"),
        "P_threshold": threshold["P_threshold"],
        **({"effective_powers": effective_powers} if is_sweep else {}),
        **({"power_definitions": power_definitions} if power_definitions else {}),
        "routines": routines_summary,
    }
    summary_path = os.path.join(run_dir, "results_summary.json")
    atomic_write_json(summary_path, summary)
    print(f"\n  Summary: {summary_path}")

    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    probe5_scenarios = [
        meta.get("scenario")
        for meta in sweep_metas
        if (meta.get("sweep") or {}).get("mode") == "probe5"
    ]
    probe5_scenarios = [name for name in probe5_scenarios if name]
    all_probe5 = len(probe5_scenarios) == len(scenarios)

    if all_probe5:
        print("  Skipping generic comparison plot for probe5 sweep.")
    else:
        print("  Generating cross-scenario comparison plot ...")
        try:
            generate_summary(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: summary plot failed: {e}", file=sys.stderr)

    if sweep_metas and not all_probe5:
        print("  Generating parameter-sweep heatmaps ...")
        try:
            generate_sweep_heatmaps(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: sweep heatmaps failed: {e}", file=sys.stderr)
        print("  Generating parameter-sweep diagnostics ...")
        try:
            generate_sweep_diagnostics(scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: sweep diagnostics failed: {e}", file=sys.stderr)

    fringe_sidecars = [
        name for name in scenarios
        if os.path.isfile(os.path.join(run_dir, f"{name}_fringe.json"))
    ]
    if fringe_sidecars:
        print("  Generating fringe summary ...")
        try:
            _generate_fringe_summary(fringe_sidecars, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: fringe summary failed: {e}", file=sys.stderr)

    if probe5_scenarios:
        print("  Generating probe5 threshold summary ...")
        try:
            _generate_probe5_summary(probe5_scenarios, run_dir, results_dir)
        except Exception as e:
            print(f"  WARNING: probe5 summary failed: {e}", file=sys.stderr)

    try:
        set_manifest_field(run_dir, "finalize_complete", True)
        set_manifest_field(run_dir, "summary_path", summary_path)
    except Exception as e:
        print(f"  WARNING: could not update manifest: {e}", file=sys.stderr)

    print("\n  Finalize complete.")


if __name__ == "__main__":
    main()
