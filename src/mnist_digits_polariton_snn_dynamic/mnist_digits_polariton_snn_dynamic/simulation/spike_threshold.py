"""CPU-only analytic spike-threshold evaluation for pulsed reservoirs.

The threshold is the pump power which maximizes upward crossings of
``s(t) = R n_R(t) - gamma_C``.  With zero reservoir diffusion the
quadratic-double reservoir is pointwise, so an isolated spot centre with
``|psi|^2 = 0`` obeys exactly

``dn_I/dt = P_c(t) - kappa n_I^2 - gamma_I n_I``
``dn_R/dt = kappa n_I^2 - gamma_R n_R``.

Thus the zero-dimensional reduction is not an approximation before the
first condensate event.  The default ``pump_only`` model omits depletion,
so its ``n_R`` is an upper bound on the physical active-reservoir density
after condensation.  The optional coupled model additionally evolves

``dn_c/dt = (R n_R - gamma_C) n_c + R_sp``
``dn_R/dt = kappa n_I^2 - gamma_R n_R - R n_R n_c``.

All calculations use NumPy float64 and do not configure or import GPU
simulation components.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import math
from pathlib import Path
import sys
from typing import Callable, Literal

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import (
    ComputeEngineParameters,
    LaserParameters,
    PhysicsConstants,
)
from polarism.grid.create_grid import create_grid
from polarism.laser.pulse_gaussian import PulseGaussian

from mnist_digits_polariton_snn_dynamic.config.loader import (
    PulseConfig,
    load_polarism_config,
    load_snn_dynamic_config,
)
from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json
from mnist_digits_polariton_snn_dynamic.scenarios.stage_meta import sha256_file


@dataclass(frozen=True, slots=True)
class SpikeThresholdSettings:
    """Settings controlling the analytic threshold scan."""

    p_min: float = 1.0
    p_max: float = 4000.0
    n_points: int = 96
    scale: Literal["log", "linear"] = "log"
    window_start_ps: float = 0.0
    window_end_ps: float | None = None
    dt_eval_ps: float | None = None
    hysteresis_rel: float = 0.02
    min_above_ps: float = 0.0
    edge_tol_rel: float = 1.0e-3
    model: Literal["pump_only", "coupled"] = "pump_only"
    spontaneous_source: float = 1.0e-6
    make_plot: bool = True


@dataclass(frozen=True, slots=True)
class CrossingResult:
    """Crossing diagnostics evaluated at one pump power."""

    n_crossings: int
    crossing_times_ps: tuple[float, ...]
    nR_max: float
    duty_above: float
    first_crossing_ps: float | None


@dataclass(frozen=True, slots=True)
class SpikeThresholdResult:
    """Complete, serializable result of a spike-threshold scan."""

    status: str
    scenario_id: str
    config_path: str
    config_sha256: str
    model: str
    reservoir_type: str
    gain_loss_definition: str
    nR_crit: float
    physics: dict[str, float]
    spot: dict[str, float]
    pulse: dict[str, float | int | str]
    window_ps: tuple[float, float]
    dt_eval_ps: float
    hysteresis_rel: float
    scan: dict[str, float | int | str]
    curve: tuple[dict[str, float | int | None], ...]
    n_crossings_max: int
    plateau: dict[str, float]
    P_threshold: float | None
    final_power_max: float | None
    crossing_times_ps: tuple[float, ...]
    sensitivity: dict[str, float | str]


class NoSpikingRegimeError(RuntimeError):
    """Raised when a scan contains no multi-spike plateau."""

    def __init__(self, result: SpikeThresholdResult) -> None:
        super().__init__("no_spiking_regime: maximum crossing count is at most one")
        self.result = result


def integrate_zero_dim(
    t: np.ndarray,
    pump: np.ndarray,
    physics: PhysicsConstants,
    *,
    model: Literal["pump_only", "coupled"] = "pump_only",
    spontaneous_source: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the zero-dimensional reservoir equations using RK4.

    Parameters
    ----------
    t, pump
        Matching time and central-pump arrays.
    physics
        Polarism physics constants.
    model
        ``pump_only`` or condensate-depleting ``coupled`` equations.
    spontaneous_source
        Source term ``R_sp`` used only by the coupled model.
    """
    times = np.asarray(t, dtype=np.float64)
    source = np.asarray(pump, dtype=np.float64)
    if times.ndim != 1 or source.shape != times.shape or times.size < 2:
        raise ValueError("t and pump must be matching one-dimensional arrays of length >= 2")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(source)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("t must be strictly increasing and t/pump must be finite")
    if model not in {"pump_only", "coupled"}:
        raise ValueError(f"Unsupported threshold model: {model!r}")
    n_r = np.zeros_like(times)
    n_i = np.zeros_like(times)
    n_c = np.full_like(times, float(physics.init_eps) ** 2) if model == "coupled" else None
    kappa, gamma_r, gamma_i, scattering, gamma_c = (float(physics.kappa), float(physics.gamma_R), float(physics.gamma_I), float(physics.R), float(physics.gamma_C))
    active = inactive = 0.0
    condensate = float(physics.init_eps) ** 2
    for index, step in enumerate(np.diff(times)):
        p0, p1 = float(source[index]), float(source[index + 1])
        pm = 0.5 * (p0 + p1)
        if model == "pump_only":
            def derivative(ar: float, ir: float, pv: float) -> tuple[float, float]:
                transfer = kappa * ir * ir
                return transfer - gamma_r * ar, pv - transfer - gamma_i * ir
            a1, i1 = derivative(active, inactive, p0)
            a2, i2 = derivative(active + 0.5 * step * a1, inactive + 0.5 * step * i1, pm)
            a3, i3 = derivative(active + 0.5 * step * a2, inactive + 0.5 * step * i2, pm)
            a4, i4 = derivative(active + step * a3, inactive + step * i3, p1)
            active = max(0.0, active + step * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0)
            inactive = max(0.0, inactive + step * (i1 + 2.0 * i2 + 2.0 * i3 + i4) / 6.0)
        else:
            def derivative(ar: float, ir: float, cr: float, pv: float) -> tuple[float, float, float]:
                transfer = kappa * ir * ir
                return transfer - gamma_r * ar - scattering * ar * cr, pv - transfer - gamma_i * ir, (scattering * ar - gamma_c) * cr + spontaneous_source
            a1, i1, c1 = derivative(active, inactive, condensate, p0)
            a2, i2, c2 = derivative(active + 0.5 * step * a1, inactive + 0.5 * step * i1, condensate + 0.5 * step * c1, pm)
            a3, i3, c3 = derivative(active + 0.5 * step * a2, inactive + 0.5 * step * i2, condensate + 0.5 * step * c2, pm)
            a4, i4, c4 = derivative(active + step * a3, inactive + step * i3, condensate + step * c3, p1)
            active = max(0.0, active + step * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0)
            inactive = max(0.0, inactive + step * (i1 + 2.0 * i2 + 2.0 * i3 + i4) / 6.0)
            condensate = max(0.0, condensate + step * (c1 + 2.0 * c2 + 2.0 * c3 + c4) / 6.0)
        n_r[index + 1], n_i[index + 1] = active, inactive
        if n_c is not None:
            n_c[index + 1] = condensate
    if n_c is None:
        return times, n_r
    return times, n_r, n_c


def count_upward_crossings(
    t: np.ndarray,
    s: np.ndarray,
    *,
    hysteresis: float,
    min_above_ps: float = 0.0,
) -> CrossingResult:
    """Count Schmitt-triggered upward gain crossings with linear timing."""
    times = np.asarray(t, dtype=np.float64)
    signal = np.asarray(s, dtype=np.float64)
    if times.ndim != 1 or signal.shape != times.shape or times.size == 0:
        raise ValueError("t and s must be matching nonempty one-dimensional arrays")
    if hysteresis < 0.0 or min_above_ps < 0.0:
        raise ValueError("hysteresis and min_above_ps must be nonnegative")
    above_duration = _positive_duration(times, signal)
    if times.size == 1:
        crossings = (float(times[0]),) if signal[0] >= hysteresis else ()
        return CrossingResult(len(crossings), crossings, float("nan"), 1.0 if signal[0] > 0 else 0.0, crossings[0] if crossings else None)
    armed = True
    candidates: list[float] = []
    if signal[0] >= hysteresis:
        candidates.append(float(times[0]))
        armed = False
    for index in range(1, times.size):
        left, right = signal[index - 1], signal[index]
        if armed and left < hysteresis <= right:
            candidates.append(_interpolate_time(times[index - 1], times[index], left, right, hysteresis))
            armed = False
        elif not armed and right <= -hysteresis:
            armed = True
    crossings = tuple(candidate for candidate in candidates if _above_run_duration(times, signal, candidate) >= min_above_ps)
    duration = max(float(times[-1] - times[0]), 0.0)
    duty = above_duration / duration if duration else float(signal[0] > 0.0)
    return CrossingResult(len(crossings), crossings, float("nan"), duty, crossings[0] if crossings else None)


def evaluate_power(
    power: float,
    t: np.ndarray,
    normalized_pump: np.ndarray,
    physics: PhysicsConstants,
    *,
    window_start_ps: float,
    window_end_ps: float,
    hysteresis_rel: float,
    min_above_ps: float,
    model: Literal["pump_only", "coupled"] = "pump_only",
    spontaneous_source: float = 1.0e-6,
) -> CrossingResult:
    """Integrate and measure one central-spot pump power."""
    integrated = integrate_zero_dim(t, float(power) * normalized_pump, physics, model=model, spontaneous_source=spontaneous_source)
    n_r = integrated[1]
    mask = (t >= window_start_ps) & (t <= window_end_ps)
    window_t = t[mask]
    window_n_r = n_r[mask]
    signal = float(physics.R) * window_n_r - float(physics.gamma_C)
    crossings = count_upward_crossings(window_t, signal, hysteresis=hysteresis_rel * float(physics.gamma_C), min_above_ps=min_above_ps)
    return replace(crossings, nR_max=float(np.max(window_n_r)))


def select_threshold_power(
    powers: np.ndarray,
    counts: np.ndarray,
    evaluate: Callable[[float], CrossingResult | int],
    *,
    edge_tol_rel: float,
) -> tuple[float, float, float, int]:
    """Select the widest maximum-count plateau and bisect its edges."""
    scanned = np.asarray(powers, dtype=np.float64)
    values = np.asarray(counts, dtype=int)
    if scanned.ndim != 1 or values.shape != scanned.shape or scanned.size < 2:
        raise ValueError("powers and counts must be matching arrays with at least two points")
    if np.any(np.diff(scanned) <= 0.0) or edge_tol_rel <= 0.0:
        raise ValueError("powers must increase and edge_tol_rel must be positive")
    maximum = int(values.max())
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value == maximum and start is None:
            start = index
        if start is not None and (value != maximum or index == values.size - 1):
            runs.append((start, index - 1 if value != maximum else index))
            start = None
    chosen = min(runs, key=lambda run: (-(scanned[run[1]] / scanned[run[0]]), scanned[run[0]]))

    def count_at(value: float) -> int:
        result = evaluate(float(value))
        return result if isinstance(result, int) else result.n_crossings

    def edge(outside: float, inside: float, keep_inside_high: bool) -> float:
        low, high = outside, inside
        while high / low - 1.0 > edge_tol_rel:
            middle = math.sqrt(low * high)
            is_maximum = count_at(middle) == maximum
            if keep_inside_high:
                if is_maximum:
                    high = middle
                else:
                    low = middle
            elif is_maximum:
                low = middle
            else:
                high = middle
        return high if keep_inside_high else low

    left, right = chosen
    p_lo = edge(scanned[left - 1], scanned[left], True) if left else float(scanned[0])
    p_hi = edge(scanned[right], scanned[right + 1], False) if right + 1 < scanned.size else float(scanned[-1])
    return p_lo, p_hi, math.sqrt(p_lo * p_hi), maximum


def find_spike_threshold(
    config_path: str,
    scenario_id: str,
    output_dir: str,
    settings: SpikeThresholdSettings | None = None,
) -> SpikeThresholdResult:
    """Find, write, and return the analytic isolated-spot spike threshold."""
    dynamic = load_snn_dynamic_config(config_path)
    cfg = load_polarism_config(dynamic.polarism_config_path)
    chosen = settings or _settings_from_config(dynamic.threshold)
    _validate_settings(chosen)
    if cfg.reservoir.reservoir_type != "quadratic-double":
        raise NotImplementedError(f"Analytic threshold does not support reservoir_type={cfg.reservoir.reservoir_type!r}")
    if float(cfg.physics.reservoir_diffusion_I) != 0.0 or float(cfg.physics.reservoir_diffusion_R) != 0.0:
        raise ValueError("reservoir_diffusion_I and reservoir_diffusion_R must both be 0.0 for exact 0-D threshold evaluation")
    stop = float(cfg.solver.total_time) if chosen.window_end_ps is None else float(chosen.window_end_ps)
    dt = float(cfg.solver.dt) if chosen.dt_eval_ps is None else float(chosen.dt_eval_ps)
    if not (0.0 <= chosen.window_start_ps < stop and dt > 0.0):
        raise ValueError("threshold window must satisfy 0 <= start < end and dt_eval_ps > 0")
    t = np.arange(0.0, stop + 0.5 * dt, dt, dtype=np.float64)
    normalized, center_value = _normalized_central_pump(t, cfg, dynamic.pulse, dynamic.geometry.sigma_space_um)
    powers = np.geomspace(chosen.p_min, chosen.p_max, chosen.n_points) if chosen.scale == "log" else np.linspace(chosen.p_min, chosen.p_max, chosen.n_points)
    cache: dict[float, CrossingResult] = {}

    def measure(power: float) -> CrossingResult:
        key = float(power)
        if key not in cache:
            cache[key] = evaluate_power(key, t, normalized, cfg.physics, window_start_ps=float(chosen.window_start_ps), window_end_ps=stop, hysteresis_rel=float(chosen.hysteresis_rel), min_above_ps=float(chosen.min_above_ps), model=chosen.model, spontaneous_source=float(chosen.spontaneous_source))
        return cache[key]

    points = [measure(float(power)) for power in powers]
    curve = tuple(_curve_point(float(power), point) for power, point in zip(powers, points))
    maximum = max(point.n_crossings for point in points)
    common = _result_base(config_path, scenario_id, cfg.physics, cfg.reservoir.reservoir_type, dynamic, chosen, center_value, stop, dt, curve, maximum, status="ok")
    if maximum <= 1:
        common["status"] = "no_spiking_regime"
        result = SpikeThresholdResult(**common, plateau={}, P_threshold=None, final_power_max=None, crossing_times_ps=(), sensitivity=_sensitivity(chosen))
        _write_artifacts(Path(output_dir), result, t, np.zeros_like(t), (), plot=chosen.make_plot)
        print(f"no_spiking_regime curve={curve}", file=sys.stderr, flush=True)
        raise NoSpikingRegimeError(result)
    p_lo, p_hi, threshold, n_max = select_threshold_power(powers, np.array([point.n_crossings for point in points]), measure, edge_tol_rel=chosen.edge_tol_rel)
    final = measure(threshold)
    if p_lo == float(powers[0]) or p_hi == float(powers[-1]):
        print("WARNING: selected spiking plateau touches a scan boundary", file=sys.stderr, flush=True)
    sensitivity = _sensitivity(chosen)
    if chosen.model == "coupled":
        sensitivity.update(
            _coupled_sensitivity(
                t,
                normalized,
                cfg.physics,
                chosen,
                powers,
                stop,
                threshold,
            )
        )
    result = SpikeThresholdResult(**common, plateau={"P_lo": p_lo, "P_hi": p_hi, "width_ratio": p_hi / p_lo}, P_threshold=threshold, final_power_max=threshold, crossing_times_ps=final.crossing_times_ps, sensitivity=sensitivity)
    if p_hi / p_lo < 1.05:
        print(f"WARNING: plateau_width_ratio={p_hi / p_lo:.6g} is below 1.05", file=sys.stderr, flush=True)
    overlap = result.spot["neighbor_pump_overlap"]
    if overlap > 1.0e-3:
        print(f"WARNING: neighbor_pump_overlap={overlap:.6g} exceeds 1e-3", file=sys.stderr, flush=True)
    signal = float(cfg.physics.R) * integrate_zero_dim(t, threshold * normalized, cfg.physics, model=chosen.model, spontaneous_source=chosen.spontaneous_source)[1] - float(cfg.physics.gamma_C)
    _write_artifacts(Path(output_dir), result, t, signal, final.crossing_times_ps, plot=chosen.make_plot)
    return result


def _normalized_central_pump(t: np.ndarray, cfg: object, pulse: PulseConfig, sigma_space: float) -> tuple[np.ndarray, float]:
    compute_engine.configure(ComputeEngineParameters(use_gpu=False))
    grid = create_grid(cfg.grid)
    laser = PulseGaussian(LaserParameters(mode="single", laser_type="pulse-gaussian", P0=1.0, Pmax=1.0, x0=0.0, y0=0.0, sigma_space=float(sigma_space), sigma_time=float(pulse.sigma_time), pulse_separation=float(pulse.pulse_separation), n_pulses=int(pulse.n_pulses), cutoff_sigma=float(pulse.cutoff_sigma), power_definition=str(pulse.power_definition), expose_results=False), grid.X, grid.Y, precision="double")
    iy, ix = np.unravel_index(np.argmin(np.asarray(grid.X) ** 2 + np.asarray(grid.Y) ** 2), grid.X.shape)
    center_at_peak = float(laser.get_power(grid.X, grid.Y, laser.phase)[iy, ix])
    peak_temporal = 1.0 if pulse.power_definition == "peak_amplitude" else 1.0 / float(laser.temporal_integral)
    center_spatial = center_at_peak / peak_temporal
    profile = center_spatial * _temporal_profile(t, pulse, laser.phase, laser.temporal_integral)
    return profile, center_spatial


def _temporal_profile(
    t: np.ndarray,
    pulse: PulseConfig,
    phase: float,
    temporal_integral: float | None,
) -> np.ndarray:
    indices = np.maximum(0, np.rint((t - phase) / float(pulse.pulse_separation)).astype(np.int64))
    valid = (int(pulse.n_pulses) <= 0) | (indices < int(pulse.n_pulses))
    offsets = t - indices * float(pulse.pulse_separation) - phase
    values = np.where(
        valid & (np.abs(offsets) <= float(pulse.cutoff_sigma) * float(pulse.sigma_time)),
        np.exp(-0.5 * (offsets / float(pulse.sigma_time)) ** 2),
        0.0,
    )
    if pulse.power_definition == "pulse_energy":
        if temporal_integral is None:
            raise ValueError("pulse-energy laser is missing temporal_integral")
        values = values / float(temporal_integral)
    return values.astype(np.float64, copy=False)


def _settings_from_config(config: object) -> SpikeThresholdSettings:
    return SpikeThresholdSettings(**asdict(config))


def _validate_settings(settings: SpikeThresholdSettings) -> None:
    if not (settings.p_min > 0.0 and settings.p_max >= settings.p_min and settings.n_points >= 2):
        raise ValueError("threshold requires 0 < p_min <= p_max and n_points >= 2")
    if settings.scale not in {"log", "linear"}:
        raise ValueError("threshold scale must be 'log' or 'linear'")
    if settings.model not in {"pump_only", "coupled"}:
        raise ValueError("threshold model must be 'pump_only' or 'coupled'")


def _result_base(
    config_path: str,
    scenario_id: str,
    physics: PhysicsConstants,
    reservoir_type: str,
    dynamic: object,
    settings: SpikeThresholdSettings,
    center_value: float,
    stop: float,
    dt: float,
    curve: tuple[dict[str, float | int | None], ...],
    maximum: int,
    *,
    status: str,
) -> dict[str, object]:
    overlap = math.exp(-0.5 * (float(dynamic.geometry.pitch_um) / float(dynamic.geometry.sigma_space_um)) ** 2)
    return {
        "status": status,
        "scenario_id": scenario_id,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(Path(config_path).resolve()),
        "model": settings.model,
        "reservoir_type": reservoir_type,
        "gain_loss_definition": "R*nR - gamma_C",
        "nR_crit": float(physics.gamma_C) / float(physics.R),
        "physics": {key: float(getattr(physics, key)) for key in ("R", "gamma_C", "gamma_R", "gamma_I", "kappa")},
        "spot": {"sigma_space_um": float(dynamic.geometry.sigma_space_um), "pump_center_spatial_value": center_value, "neighbor_pump_overlap": overlap},
        "pulse": {"sigma_time": float(dynamic.pulse.sigma_time), "pulse_separation": float(dynamic.pulse.pulse_separation), "n_pulses": int(dynamic.pulse.n_pulses), "cutoff_sigma": float(dynamic.pulse.cutoff_sigma), "power_definition": str(dynamic.pulse.power_definition)},
        "window_ps": (float(settings.window_start_ps), stop),
        "dt_eval_ps": dt,
        "hysteresis_rel": float(settings.hysteresis_rel),
        "scan": {"P_min": float(settings.p_min), "P_max": float(settings.p_max), "n_points": int(settings.n_points), "scale": settings.scale},
        "curve": curve,
        "n_crossings_max": maximum,
    }


def _curve_point(power: float, result: CrossingResult) -> dict[str, float | int | None]:
    return {"P": power, "n_crossings": result.n_crossings, "nR_max": result.nR_max, "duty_above": result.duty_above, "first_crossing_ps": result.first_crossing_ps}


def _sensitivity(settings: SpikeThresholdSettings) -> dict[str, float | str]:
    return {
        "spontaneous_source": float(settings.spontaneous_source),
        "interpretation": "not used by pump_only" if settings.model == "pump_only" else "threshold variation across spontaneous-source scales",
    }


def _coupled_sensitivity(
    t: np.ndarray,
    normalized: np.ndarray,
    physics: PhysicsConstants,
    settings: SpikeThresholdSettings,
    powers: np.ndarray,
    window_end_ps: float,
    nominal_threshold: float,
) -> dict[str, float | str]:
    sensitivity: dict[str, float | str] = {"threshold_at_Rsp_1x": nominal_threshold}
    for scale in (0.1, 10.0):
        source = float(settings.spontaneous_source) * scale
        points = [
            evaluate_power(
                float(power),
                t,
                normalized,
                physics,
                window_start_ps=float(settings.window_start_ps),
                window_end_ps=window_end_ps,
                hysteresis_rel=float(settings.hysteresis_rel),
                min_above_ps=float(settings.min_above_ps),
                model="coupled",
                spontaneous_source=source,
            )
            for power in powers
        ]
        maximum = max(point.n_crossings for point in points)
        key = f"threshold_at_Rsp_{scale:g}x"
        if maximum <= 1:
            sensitivity[key] = "no_spiking_regime"
            continue
        _, _, threshold, _ = select_threshold_power(
            powers,
            np.array([point.n_crossings for point in points]),
            lambda power: evaluate_power(
                power,
                t,
                normalized,
                physics,
                window_start_ps=float(settings.window_start_ps),
                window_end_ps=window_end_ps,
                hysteresis_rel=float(settings.hysteresis_rel),
                min_above_ps=float(settings.min_above_ps),
                model="coupled",
                spontaneous_source=source,
            ),
            edge_tol_rel=float(settings.edge_tol_rel),
        )
        sensitivity[key] = threshold
    return sensitivity


def _write_artifacts(output_dir: Path, result: SpikeThresholdResult, t: np.ndarray, signal: np.ndarray, crossings: tuple[float, ...], *, plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(output_dir / "spike_threshold.json"), asdict(result))
    with (output_dir / "spike_threshold_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("P", "n_crossings", "nR_max", "duty_above", "first_crossing_ps"))
        writer.writeheader()
        writer.writerows(sorted(result.curve, key=lambda item: float(item["P"])))
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        powers = np.array([float(item["P"]) for item in result.curve])
        axes[0].plot(powers, [int(item["n_crossings"]) for item in result.curve], marker="o")
        axes[0].set_xscale("log")
        axes[0].set(xlabel="P", ylabel="upward crossings")
        if result.plateau:
            axes[0].axvspan(result.plateau["P_lo"], result.plateau["P_hi"], alpha=0.2)
            axes[0].axvline(result.P_threshold, color="tab:red")
        axes[1].plot(t, signal)
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        if crossings:
            axes[1].plot(crossings, np.zeros(len(crossings)), "o")
        axes[1].set(xlabel="time (ps)", ylabel="R nR - gamma_C")
        fig.tight_layout()
        fig.savefig(output_dir / "spike_threshold.png", dpi=150)
        plt.close(fig)


def _interpolate_time(t0: float, t1: float, y0: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def _positive_duration(t: np.ndarray, s: np.ndarray) -> float:
    total = 0.0
    for t0, t1, s0, s1 in zip(t[:-1], t[1:], s[:-1], s[1:]):
        step = float(t1 - t0)
        if s0 > 0.0 and s1 > 0.0:
            total += step
        elif s0 > 0.0 >= s1:
            total += _interpolate_time(float(t0), float(t1), float(s0), float(s1), 0.0) - float(t0)
        elif s0 <= 0.0 < s1:
            total += float(t1) - _interpolate_time(float(t0), float(t1), float(s0), float(s1), 0.0)
    return total


def _above_run_duration(t: np.ndarray, s: np.ndarray, crossing: float) -> float:
    index = int(np.searchsorted(t, crossing, side="right"))
    end = float(t[-1])
    for current in range(max(index, 1), t.size):
        if s[current - 1] > 0.0 >= s[current]:
            end = _interpolate_time(float(t[current - 1]), float(t[current]), float(s[current - 1]), float(s[current]), 0.0)
            break
    return max(0.0, end - crossing)
