"""Fit algebraic convergence orders and identify spectral or floor-limited data.

The per-series floor is the larger of 1e-11 absolute error and 1e-11 times
the coarsest-series error. This relative factor is far below an observable
asymptotic decrease while rejecting values produced at double-precision and
reference-solution floors. The coarsest point is always rejected as
pre-asymptotic; the rejection is never relaxed merely to obtain the three
points required for regression. Spatial spectral convergence is identified
when at least two adjacent slopes exist and the finest slope exceeds the
coarsest by more than one order, the expected signature of
faster-than-power-law decay.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path("artifacts/convergence")
FIGURES = Path("figures")
SOLVERS = (
    "rk4-fdm",
    "rk4-fdm-fused",
    "ip-rk4",
    "ifrk4-fft-cuda",
    "split-step-fft",
    "etd-rk2",
)
SPECTRAL_SOLVERS = {"ip-rk4", "ifrk4-fft-cuda", "split-step-fft", "etd-rk2"}
RESERVOIRS = ("single", "quadratic-double")
TIME_BANDS = {
    "rk4-fdm": ([3.5, 4.5], [3.5, 4.5]),
    "rk4-fdm-fused": ([3.5, 4.5], [3.5, 4.5]),
    "ip-rk4": ([3.5, 4.5], [3.5, 4.5]),
    "ifrk4-fft-cuda": ([3.5, 4.5], [3.5, 4.5]),
    "split-step-fft": ([1.5, 2.5], [0.5, 1.5]),
    "etd-rk2": ([1.5, 2.5], [1.5, 2.5]),
}
SPACE_CASES = ("bandlimited", "broadband")
SPACE_BANDS = {
    "rk4-fdm_bandlimited": [1.5, 2.5],
    "rk4-fdm-fused_bandlimited": [1.5, 2.5],
}
NUMERICAL_ERROR_FLOOR = 1e-11
RELATIVE_FLOOR_FACTOR = 1e-11
MINIMUM_REGRESSION_POINTS = 3
SPECTRAL_SLOPE_INCREASE = 1.0


def _local_slopes(
    ordered: list[dict[str, float]], axis: str
) -> list[float | None]:
    slopes: list[float | None] = []
    for coarse, fine in zip(ordered, ordered[1:]):
        if coarse["error"] <= 0.0 or fine["error"] <= 0.0:
            slopes.append(None)
        else:
            slopes.append(
                float(
                    np.log(coarse["error"] / fine["error"])
                    / np.log(coarse[axis] / fine[axis])
                )
            )
    return slopes


def _fit(
    entries: list[dict[str, float]],
    band: list[float] | None,
    allow_spectral: bool = False,
) -> dict[str, Any]:
    axis = "dt" if "dt" in entries[0] else "dx"
    ordered = sorted(entries, key=lambda entry: entry[axis], reverse=True)
    errors = np.asarray([entry["error"] for entry in ordered], dtype=np.float64)
    floor = max(NUMERICAL_ERROR_FLOOR, RELATIVE_FLOOR_FACTOR * float(errors.max()))
    slopes = _local_slopes(ordered, axis)
    finite_slopes = [value for value in slopes if value is not None]
    is_spectral = (
        allow_spectral
        and len(finite_slopes) >= 2
        and finite_slopes[-1] > finite_slopes[0] + SPECTRAL_SLOPE_INCREASE
    )
    discarded: list[dict[str, Any]] = []
    retained: list[tuple[int, dict[str, float]]] = []
    for index, entry in enumerate(ordered):
        if entry["error"] < floor:
            discarded.append(
                {"index": index, **entry, "reason": "below-machine-precision-floor"}
            )
        elif index == 0:
            discarded.append({"index": index, **entry, "reason": "coarsest-dropped"})
        else:
            retained.append((index, entry))
    result: dict[str, Any] = {
        "order": None,
        "fit_r2": None,
        "used_indices": [index for index, _ in retained],
        "discarded": discarded,
        "expected_order_band": band,
        "reason_no_fit": None,
        "n_points_used": len(retained),
        "n_points_total": len(ordered),
        "convergence_regime": "algebraic",
        "local_slopes": slopes,
        "numerical_floor": floor,
    }
    if is_spectral:
        result["convergence_regime"] = "spectral"
        result["reason_no_fit"] = (
            "spectral (exponential) convergence — algebraic order not defined"
        )
        return result
    if np.all(errors < floor):
        result["convergence_regime"] = "floor-limited"
        result["reason_no_fit"] = "errors at numerical floor"
        return result
    if len(retained) < MINIMUM_REGRESSION_POINTS:
        result["reason_no_fit"] = (
            f"insufficient asymptotic points (n={len(retained)}, minimum "
            f"{MINIMUM_REGRESSION_POINTS})"
        )
        return result
    x = np.log([entry[axis] for _, entry in retained])
    y = np.log([entry["error"] for _, entry in retained])
    order, intercept = np.polyfit(x, y, 1)
    predicted = order * x + intercept
    total = np.sum((y - np.mean(y)) ** 2)
    result["order"] = float(order)
    result["fit_r2"] = (
        float(1.0 - np.sum((y - predicted) ** 2) / total) if total else 1.0
    )
    return result


def _missing_result(band: list[float] | None) -> dict[str, Any]:
    return {
        "order": None,
        "fit_r2": None,
        "used_indices": [],
        "discarded": [],
        "expected_order_band": band,
        "reason_no_fit": "input artefact missing",
        "n_points_used": 0,
        "n_points_total": 0,
        "convergence_regime": "floor-limited",
        "local_slopes": [],
    }


def _load_time() -> tuple[dict[str, Any], dict[str, list[dict[str, float]]]]:
    results: dict[str, Any] = {}
    series: dict[str, list[dict[str, float]]] = {}
    for solver in SOLVERS:
        for reservoir_index, reservoir in enumerate(RESERVOIRS):
            key = f"{solver}_{reservoir}"
            path = ARTIFACTS / f"time_{key}.json"
            band = TIME_BANDS[solver][reservoir_index]
            if not path.exists():
                print(f"WARNING: missing input artefact {path}")
                results[key] = _missing_result(band)
                continue
            entries = json.loads(path.read_text())["entries"]
            series[key] = entries
            results[key] = _fit(entries, band)
    return results, series


def _load_space() -> tuple[dict[str, Any], dict[str, list[dict[str, float]]]]:
    results: dict[str, Any] = {}
    series: dict[str, list[dict[str, float]]] = {}
    for solver in SOLVERS:
        for case in SPACE_CASES:
            key = f"{solver}_{case}"
            band = SPACE_BANDS.get(key)
            path = ARTIFACTS / f"space_{key}.json"
            if not path.exists():
                print(f"WARNING: missing input artefact {path}")
                results[key] = _missing_result(band)
                continue
            data = json.loads(path.read_text())
            entries = data.get("entries")
            if not isinstance(entries, list) or not entries or any(
                not isinstance(entry, dict)
                or "dx" not in entry
                or "error" not in entry
                for entry in entries
            ):
                result = _missing_result(band)
                result["reason_no_fit"] = "input entries missing"
                results[key] = result
                continue
            result = _fit(entries, band, allow_spectral=solver in SPECTRAL_SOLVERS)
            if result["order"] is not None:
                reference_nx = data.get("reference_nx")
                if isinstance(reference_nx, int) and reference_nx > 0:
                    finest_nx = max(int(entry["nx"]) for entry in entries)
                    if reference_nx / finest_nx < 3:
                        result["reference_bias_warning"] = True
            series[key] = entries
            results[key] = result
    return results, series


def _fit_label(key: str, result: dict[str, Any]) -> str:
    if result["convergence_regime"] == "spectral":
        return f"{key} (zbieżność spektralna)"
    order = result["order"]
    return f"{key} (p={order:+.2f})" if order is not None else f"{key} (no fit)"


def _plot_time(results: dict[str, Any], series: dict[str, list[dict[str, float]]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, reservoir in zip(axes, RESERVOIRS):
        for index, solver in enumerate(SOLVERS):
            key = f"{solver}_{reservoir}"
            if key not in series:
                continue
            entries = sorted(series[key], key=lambda entry: entry["dt"], reverse=True)
            axis.loglog(
                [entry["dt"] for entry in entries],
                [entry["error"] for entry in entries],
                "-o",
                color=f"C{index}",
                label=_fit_label(solver, results[key]),
            )
        axis.set_title(f"reservoir = {reservoir}")
        axis.set_xlabel("dt")
        axis.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("error")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_time_convergence.pdf")
    plt.close(fig)


def _plot_space(results: dict[str, Any], series: dict[str, list[dict[str, float]]]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 5))
    for case in SPACE_CASES:
        for index, solver in enumerate(SOLVERS):
            key = f"{solver}_{case}"
            if key not in series or (case == "broadband" and solver.startswith("rk4-fdm")):
                continue
            entries = sorted(series[key], key=lambda entry: entry["dx"], reverse=True)
            axis.loglog(
                [entry["dx"] for entry in entries],
                [entry["error"] for entry in entries],
                "-o" if case == "bandlimited" else "--o",
                color=f"C{index}",
                label=_fit_label(f"{case}: {solver}", results[key]),
            )
    axis.set_xlabel("dx")
    axis.set_ylabel("error")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_space_convergence.pdf")
    plt.close(fig)


def main() -> None:
    """Analyse existing artifacts and write fitted orders and figures."""
    time_results, time_series = _load_time()
    space_results, space_series = _load_space()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = {"time": time_results, "space": space_results}
    (ARTIFACTS / "fitted_orders.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    _plot_time(time_results, time_series)
    _plot_space(space_results, space_series)


if __name__ == "__main__":
    main()
