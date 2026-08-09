"""Phoenix benchmark tests."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import polarism.laser
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config

pytestmark = pytest.mark.slow


GRID_SIZE = 200
L_MAX = 20.0
DX = L_MAX / (GRID_SIZE - 1)

HBAR = 6.582119569e-4
M_EFF = 0.5 * 5.864e-4

GAMMA_C = 0.1
GAMMA_R = 0.15
R_COND = 0.003
G_C = 6e-6
G_R = 1.2e-5

INIT_EPS = 1e-7


def _asnumpy(x):
    """Convert a backend array to NumPy."""
    xp = compute_engine.xp
    return xp.asnumpy(x) if hasattr(xp, "asnumpy") else np.asarray(x)


def _gpu_sync():
    """Wait for queued GPU work to finish."""
    if compute_engine.use_gpu:
        try:
            import cupy

            cupy.cuda.Stream.null.synchronize()
        except Exception:
            pass


def _load_phoenix_rho_max(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load phoenix rho max."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    t = data[:, 0].astype(np.float64)
    rho = data[:, 1].astype(np.float64)
    idx = np.argsort(t)
    return t[idx], rho[idx]


def _load_phoenix_frame(path: Path) -> dict | None:
    """Load phoenix frame."""
    if not path.exists():
        return None
    data = np.load(path)
    rho = data["rho"]
    nR = data["nR"] if "nR" in data else np.zeros_like(rho)
    return {
        "t": float(data["t"]),
        "psi": data["psi_real"] + 1j * data["psi_imag"],
        "rho": rho,
        "nR": nR,
    }


def _load_phoenix_timing(path: Path) -> dict | None:
    """Load phoenix timing."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_fig(outdir: Path, name: str) -> None:
    """Save the current figure."""
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / name, dpi=160, bbox_inches="tight")
    plt.close()


def compute_accuracy_metrics(
    t_ref: np.ndarray,
    rho_ref: np.ndarray,
    t_sim: np.ndarray,
    rho_sim: np.ndarray,
    tail_fraction: float = 0.1,
) -> dict:
    """Compute accuracy metrics."""
    rho_sim_interp = np.interp(t_ref, t_sim, rho_sim)
    floor = max(1e-20, 1e-8 * float(rho_ref.max()))

    final_rel = abs(rho_sim_interp[-1] - rho_ref[-1]) / max(abs(rho_ref[-1]), floor)

    log_ref = np.log10(np.maximum(rho_ref, floor))
    log_sim = np.log10(np.maximum(rho_sim_interp, floor))
    log_rmse = float(np.sqrt(np.mean((log_sim - log_ref) ** 2)))

    tail_threshold = tail_fraction * float(rho_ref.max())
    tail_mask = rho_ref >= tail_threshold
    if np.any(tail_mask):
        tail_rel_errors = np.abs(
            rho_sim_interp[tail_mask] - rho_ref[tail_mask]
        ) / np.maximum(rho_ref[tail_mask], floor)
        max_tail_rel = float(tail_rel_errors.max())
        mean_tail_rel = float(tail_rel_errors.mean())
    else:
        max_tail_rel = np.nan
        mean_tail_rel = np.nan

    thresholds = [1e-3, 1e-2, 1e-1, 0.5, 0.9]
    rho_max_val = float(rho_ref.max())
    threshold_dt_errors = []
    for frac in thresholds:
        level = frac * rho_max_val
        t_ref_cross = _first_crossing_time(t_ref, rho_ref, level)
        t_sim_cross = _first_crossing_time(t_ref, rho_sim_interp, level)
        if np.isfinite(t_ref_cross) and np.isfinite(t_sim_cross):
            threshold_dt_errors.append(abs(t_sim_cross - t_ref_cross))
    crossing_dt_max = max(threshold_dt_errors) if threshold_dt_errors else 0.0

    if len(log_ref) > 1:
        correlation = float(np.corrcoef(log_ref, log_sim)[0, 1])
    else:
        correlation = np.nan

    return {
        "final_rel_error": final_rel,
        "log_rmse": log_rmse,
        "max_tail_rel_error": max_tail_rel,
        "mean_tail_rel_error": mean_tail_rel,
        "crossing_dt_max": crossing_dt_max,
        "correlation": correlation,
    }


def _first_crossing_time(t: np.ndarray, y: np.ndarray, level: float) -> float:
    """Return the first time the signal crosses the level."""
    idx = np.where(y >= level)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return t1
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0)


def configure_simulation(
    cfg: Config,
    lasers_yaml: Path,
    solver_name: str,
    total_time: float,
    use_gpu: bool = None,
) -> None:
    """Configure simulation."""
    if use_gpu is not None:
        cfg.compute_engine.use_gpu = use_gpu

    L_periodic = L_MAX / (GRID_SIZE - 1) * GRID_SIZE
    cfg.grid.lx = L_periodic
    cfg.grid.ly = L_periodic
    cfg.grid.nx = GRID_SIZE
    cfg.grid.ny = GRID_SIZE
    cfg.grid.grid_type = "periodic"

    cfg.boundary_condition.absorption = "no-absorption"
    cfg.boundary_condition.strength = 0.0

    cfg.reservoir.reservoir_type = "single"

    cfg.physics.hbar = HBAR
    cfg.physics.m_eff = M_EFF
    cfg.physics.gamma_C = GAMMA_C
    cfg.physics.gamma_R = GAMMA_R
    cfg.physics.R = R_COND
    cfg.physics.g_C = G_C
    cfg.physics.g_R = G_R
    cfg.physics.init_eps = INIT_EPS

    cfg.solver.method = solver_name
    cfg.solver.dt = 0.001
    cfg.solver.total_time = float(total_time)
    cfg.solver.precision = "double"

    cfg.laser.mode = "multiple"
    cfg.laser.config_file = str(lasers_yaml)

    cfg.result.real_time_view = False
    cfg.result.save_results = False
    cfg.result.save_hdf5 = False
    cfg.result.save_json = False
    cfg.result.save_npy = False


def apply_potential_config(cfg: Config, case_name: str) -> None:
    """Apply potential config."""
    if case_name == "01_uniform_pump":
        cfg.potential.potential_type = "zero"
    elif case_name == "02_pump_no_potential":
        cfg.potential.potential_type = "zero"
    elif case_name == "03_pump_and_potential":
        cfg.potential.potential_type = "double-well-supergaussian"
        cfg.potential.x1, cfg.potential.y1 = -3.0, 0.0
        cfg.potential.x2, cfg.potential.y2 = 3.0, 0.0
        cfg.potential.V1, cfg.potential.V2 = -0.3, -0.35
        cfg.potential.w1, cfg.potential.w2 = 4.0, 4.0
        cfg.potential.order = 1.0
    else:
        raise ValueError(f"Unknown case: {case_name}")


def _phoenix_grid():
    """Return the benchmark grid settings."""
    x = np.linspace(-L_MAX / 2, L_MAX / 2, GRID_SIZE, endpoint=True)
    y = np.linspace(-L_MAX / 2, L_MAX / 2, GRID_SIZE, endpoint=True)
    return np.meshgrid(x, y, indexing="xy")


def _reconstruct_pump(lasers_yaml: Path, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rebuild the pump field from the laser YAML file."""
    import yaml

    with open(lasers_yaml) as f:
        data = yaml.safe_load(f)

    pump = np.zeros_like(X)
    for item in data["lasers"]:
        P0 = item["P0"]
        x0 = item.get("x0", 0.0)
        y0 = item.get("y0", 0.0)
        ltype = item["laser_type"]
        sigma = item.get("sigma_space", 1.0)

        r2 = (X - x0) ** 2 + (Y - y0) ** 2
        if ltype == "uniform":
            pump += P0 * np.ones_like(X)
        elif ltype == "continuous-gaussian":
            pump += P0 * np.exp(-0.5 * r2 / sigma**2)
        elif ltype == "continuous-exp":
            pump += P0 * np.exp(-np.sqrt(r2) / sigma**2)
        else:
            raise ValueError(f"Unknown laser type in YAML: {ltype}")
    return pump


def _reconstruct_potential(
    case_name: str, X: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """Rebuild the benchmark potential for the case."""
    if case_name in ("01_uniform_pump", "02_pump_no_potential"):
        return np.zeros_like(X)
    if case_name == "03_pump_and_potential":
        x1, y1 = -3.0, 0.0
        x2, y2 = 3.0, 0.0
        V1, V2 = -0.3, -0.35
        w1, w2 = 4.0, 4.0
        order = 1.0
        r2_1 = (X - x1) ** 2 + (Y - y1) ** 2
        r2_2 = (X - x2) ** 2 + (Y - y2) ** 2
        return V1 * np.exp(-((r2_1 / w1**2) ** order)) + V2 * np.exp(
            -((r2_2 / w2**2) ** order)
        )
    raise ValueError(f"Unknown case: {case_name}")


def validate_benchmark_data(case_name: str, case_dir: Path, lasers_yaml: Path) -> None:
    """Validate benchmark data."""
    psi_path = case_dir / "psi_init.txt"
    assert psi_path.exists(), f"Missing psi_init.txt in {case_dir}"
    psi_data = np.loadtxt(psi_path, comments="#")
    assert psi_data.shape == (2 * GRID_SIZE, GRID_SIZE), (
        f"psi_init.txt shape {psi_data.shape} != expected "
        f"({2 * GRID_SIZE}, {GRID_SIZE})"
    )

    pump_path = case_dir / "pump.txt"
    assert pump_path.exists(), f"Missing pump.txt in {case_dir}"
    pump_file = np.loadtxt(pump_path, comments="#")
    assert pump_file.shape == (GRID_SIZE, GRID_SIZE), (
        f"pump.txt shape {pump_file.shape} != expected "
        f"({GRID_SIZE}, {GRID_SIZE})"
    )

    pot_path = case_dir / "potential.txt"
    assert pot_path.exists(), f"Missing potential.txt in {case_dir}"
    pot_file = np.loadtxt(pot_path, comments="#")
    assert pot_file.shape == (GRID_SIZE, GRID_SIZE), (
        f"potential.txt shape {pot_file.shape} != expected "
        f"({GRID_SIZE}, {GRID_SIZE})"
    )


def validate_simulation_inputs(
    sim, case_dir: Path, case_name: str, atol: float = 1e-10
) -> None:
    """Validate simulation inputs."""
    xp = compute_engine.xp

    X_ph, Y_ph = _phoenix_grid()
    X_pol = _asnumpy(sim.grid.X)
    Y_pol = _asnumpy(sim.grid.Y)
    grid_max_diff = max(
        float(np.max(np.abs(X_pol - X_ph))),
        float(np.max(np.abs(Y_pol - Y_ph))),
    )
    assert grid_max_diff < atol, (
        f"Grid mismatch: max coordinate diff = {grid_max_diff:.2e}. "
        f"Polarism grid does not match Phoenix grid."
    )

    pump_path = case_dir / "pump.txt"
    pump_phoenix = np.loadtxt(pump_path, comments="#")
    pump_polarism = _asnumpy(sim._compute_total_pump(0.0))
    pump_max_diff = float(np.max(np.abs(pump_polarism - pump_phoenix)))
    pump_scale = max(float(pump_phoenix.max()), 1e-30)
    pump_rel = pump_max_diff / pump_scale
    assert pump_rel < 1e-6, (
        f"Pump mismatch: max abs diff = {pump_max_diff:.2e}, "
        f"max rel diff = {pump_rel:.2e}. "
        f"Polarism pump differs from Phoenix pump.txt."
    )

    pot_path = case_dir / "potential.txt"
    pot_phoenix = np.loadtxt(pot_path, comments="#")
    pot_polarism = _asnumpy(sim.potential)
    if np.iscomplexobj(pot_polarism):
        pot_polarism = pot_polarism.real
    pot_max_diff = float(np.max(np.abs(pot_polarism - pot_phoenix)))
    pot_scale = max(float(np.abs(pot_phoenix).max()), 1e-30)
    if pot_scale > 1e-20:
        pot_rel = pot_max_diff / pot_scale
        assert pot_rel < 1e-6, (
            f"Potential mismatch: max abs diff = {pot_max_diff:.2e}, "
            f"max rel diff = {pot_rel:.2e}. "
            f"Polarism potential differs from Phoenix potential.txt."
        )
    else:
        assert pot_max_diff < atol, (
            f"Potential should be zero but max diff = {pot_max_diff:.2e}"
        )

    psi_path = case_dir / "psi_init.txt"
    psi_data = np.loadtxt(psi_path, comments="#")
    N = psi_data.shape[0] // 2
    psi_phoenix = psi_data[:N, :] + 1j * psi_data[N:, :]
    psi_polarism = _asnumpy(sim.state.psi)
    psi_max_diff = float(np.max(np.abs(psi_polarism - psi_phoenix)))
    assert psi_max_diff < atol, (
        f"psi_init mismatch: max diff = {psi_max_diff:.2e}. "
        f"Polarism initial state differs from Phoenix psi_init.txt."
    )


def _load_psi_init(path: Path, psi_dtype) -> np.ndarray | None:
    """Load psi init."""
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    N = data.shape[0] // 2
    psi = (data[:N, :] + 1j * data[N:, :]).astype(psi_dtype)
    return psi


def run_benchmark_simulation(
    cfg: Config,
    t_ref: np.ndarray,
    psi_init_path: Path | None = None,
    sample_dt: float = 0.2,
    save_frames: bool = False,
    case_dir: Path | None = None,
    case_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, dict]:
    """Run benchmark simulation."""
    from polarism.simulation_controller import SimulationController

    xp = compute_engine.xp

    gc.collect()

    sim = SimulationController(cfg)

    P0 = sim._compute_total_pump(0.0)

    if cfg.solver.precision == "single":
        psi_dtype = np.complex64
        real_dtype = np.float32
    else:
        psi_dtype = np.complex128
        real_dtype = np.float64

    P0 = P0.astype(real_dtype)
    if xp.iscomplexobj(sim.potential):
        sim.potential = sim.potential.astype(psi_dtype)
    else:
        sim.potential = sim.potential.astype(real_dtype)

    z = xp.zeros(P0.shape, dtype=real_dtype)
    if hasattr(sim.reservoir, "set_state"):
        sim.reservoir.set_state((z,))
    elif hasattr(sim.reservoir, "nR"):
        sim.reservoir.nR = z

    psi0 = None
    if psi_init_path is not None:
        psi0 = _load_psi_init(psi_init_path, psi_dtype)

    if psi0 is not None:
        sim.state.psi = xp.asarray(psi0)
    else:
        sim.state.psi = xp.asarray(
            (INIT_EPS * np.ones_like(_asnumpy(P0))).astype(psi_dtype)
        )

    if case_dir is not None and case_name is not None:
        validate_simulation_inputs(sim, case_dir, case_name)

    dt = float(cfg.solver.dt)
    t_end = float(t_ref[-1])
    n_steps = int(np.ceil(t_end / dt))

    sample_steps = set()
    for t_val in t_ref:
        if t_val <= 0.0:
            continue
        s = int(round(t_val / dt))
        if 1 <= s <= n_steps:
            sample_steps.add(s)
    sample_steps.add(n_steps)

    times = []
    rho_max = []
    frames = {"first": None, "last": None}

    if save_frames:
        frames["first"] = {
            "t": 0.0,
            "psi": _asnumpy(sim.state.psi).copy(),
            "rho": _asnumpy(xp.abs(sim.state.psi) ** 2),
            "nR": _asnumpy(sim.reservoir.get_reservoir_density()),
        }

    rho_init = xp.abs(sim.state.psi) ** 2
    times.append(0.0)
    rho_max.append(float(rho_init.max()))

    _gpu_sync()

    start_time = time.perf_counter()

    for step in range(n_steps):
        sim.state.t = step * dt
        sim.solver.step(
            sim.potential, P0, sim.reservoir, sim.boundary_condition, sim.state
        )

        if (step + 1) in sample_steps:
            t_now = (step + 1) * dt
            rho_now = xp.abs(sim.state.psi) ** 2
            times.append(t_now)
            rho_max.append(float(rho_now.max()))

    _gpu_sync()
    elapsed = time.perf_counter() - start_time

    if save_frames:
        frames["last"] = {
            "t": t_end,
            "psi": _asnumpy(sim.state.psi).copy(),
            "rho": _asnumpy(xp.abs(sim.state.psi) ** 2),
            "nR": _asnumpy(sim.reservoir.get_reservoir_density()),
        }

    peak_mb = 0.0

    return np.asarray(times), np.asarray(rho_max), elapsed, peak_mb, frames


def plot_benchmark_results(
    outdir: Path,
    case_name: str,
    solver_name: str,
    t_ref: np.ndarray,
    rho_ref: np.ndarray,
    t_sim: np.ndarray,
    rho_sim: np.ndarray,
    metrics: dict,
    phoenix_timing: dict = None,
) -> None:
    """Plot benchmark results."""
    outdir.mkdir(parents=True, exist_ok=True)

    rho_sim_interp = np.interp(t_ref, t_sim, rho_sim)
    floor = max(1e-20, 1e-8 * float(rho_ref.max()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(t_ref, rho_ref, "b-", label="PHOENIX", linewidth=1.5)
    axes[0].plot(
        t_sim, rho_sim, "r--", label=f"Polarism ({solver_name})", linewidth=1.2
    )
    axes[0].set_xlabel("t (ps)")
    axes[0].set_ylabel(r"max($|\psi|^2$)")
    axes[0].set_title(f"{case_name}: Density Evolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(t_ref, rho_ref, "b-", label="PHOENIX", linewidth=1.5)
    axes[1].semilogy(
        t_sim, rho_sim, "r--", label=f"Polarism ({solver_name})", linewidth=1.2
    )
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylabel(r"max($|\psi|^2$)")
    axes[1].set_title(f"{case_name}: Log Scale")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    _save_fig(outdir, "rho_max_compare.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    log_ref = np.log10(np.maximum(rho_ref, floor))
    log_sim = np.log10(np.maximum(rho_sim_interp, floor))
    axes[0].plot(t_ref, log_sim - log_ref, "k-", linewidth=1)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("t (ps)")
    axes[0].set_ylabel(r"$\log_{10}(\rho_{sim}) - \log_{10}(\rho_{ref})$")
    axes[0].set_title("Log-space Error")
    axes[0].grid(True, alpha=0.3)

    rel_error = np.abs(rho_sim_interp - rho_ref) / np.maximum(rho_ref, floor)
    axes[1].semilogy(t_ref, rel_error, "k-", linewidth=1)
    axes[1].set_xlabel("t (ps)")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_title("Relative Error vs Time")
    axes[1].grid(True, alpha=0.3)

    _save_fig(outdir, "error_analysis.png")

    if phoenix_timing:
        fig, ax = plt.subplots(figsize=(8, 5))

        polarism_time = metrics.get("elapsed_time_s", 0)
        phoenix_time = phoenix_timing.get("elapsed_s", 0)
        t_end_phoenix = phoenix_timing.get("t_end_ps", 500.0)
        t_end_sim = metrics.get("sim_time_ps", t_end_phoenix)

        device_polarism = "GPU" if compute_engine.use_gpu else "CPU"
        device_phoenix = phoenix_timing.get("device", "Unknown")

        if t_end_phoenix > 0:
            phoenix_scaled = phoenix_time * (t_end_sim / t_end_phoenix)
        else:
            phoenix_scaled = phoenix_time

        prec_phoenix = phoenix_timing.get("precision", "unknown")
        bars = ax.bar(
            [
                f"Polarism ({solver_name})\n({device_polarism}, fp64)",
                f"PHOENIX\n({device_phoenix}, {prec_phoenix})",
            ],
            [polarism_time, phoenix_scaled],
            color=["#2ecc71", "#3498db"],
            edgecolor="black",
        )

        ax.set_ylabel("Execution Time (s)")
        ax.set_title(f"{case_name}: Performance ({t_end_sim:.0f} ps)")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, t in zip(bars, [polarism_time, phoenix_scaled]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.2f}s",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        if phoenix_scaled > 0 and polarism_time > 0:
            speedup = phoenix_scaled / polarism_time
            label = (
                f"Speedup: {speedup:.2f}x"
                if speedup >= 1
                else f"Slowdown: {1/speedup:.2f}x"
            )
            ax.text(
                0.95,
                0.95,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        _save_fig(outdir, "performance_compare.png")

    device = "GPU" if compute_engine.use_gpu else "CPU"
    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"Benchmark: {case_name}\n")
        f.write(f"Solver: {solver_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Grid: {GRID_SIZE} x {GRID_SIZE}\n")
        f.write(f"dt: {0.001} ps\n")
        f.write("-" * 40 + "\n")
        f.write("ACCURACY METRICS:\n")
        for key, val in metrics.items():
            if (
                key.startswith("elapsed")
                or key.startswith("peak")
                or key.startswith("steps")
                or key.startswith("sim_time")
            ):
                continue
            f.write(f"  {key}: {val:.6e}\n")
        f.write("-" * 40 + "\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"  elapsed_time_s: {metrics.get('elapsed_time_s', 0):.3f}\n")
        f.write(f"  peak_memory_mb: {metrics.get('peak_memory_mb', 0):.2f}\n")
        f.write(f"  steps_per_second: {metrics.get('steps_per_second', 0):.1f}\n")

        if phoenix_timing:
            f.write("-" * 40 + "\n")
            f.write("PHOENIX REFERENCE:\n")
            f.write(f"  device: {phoenix_timing.get('device', 'Unknown')}\n")
            f.write(f"  elapsed_s: {phoenix_timing.get('elapsed_s', 0):.3f}\n")
            f.write(f"  precision: {phoenix_timing.get('precision', 'Unknown')}\n")


def plot_frame_comparison(
    outdir: Path,
    case_name: str,
    frame_name: str,
    phoenix_frame: dict,
    polarism_frame: dict,
) -> None:
    """Plot frame comparison."""
    if phoenix_frame is None or polarism_frame is None:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    extent = [-L_MAX / 2, L_MAX / 2, -L_MAX / 2, L_MAX / 2]

    im0 = axes[0, 0].imshow(phoenix_frame["rho"], origin="lower", extent=extent)
    axes[0, 0].set_title(f"PHOENIX: rho (t={phoenix_frame['t']:.1f} ps)")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(phoenix_frame["nR"], origin="lower", extent=extent)
    axes[0, 1].set_title("PHOENIX: nR")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(
        np.angle(phoenix_frame["psi"]), origin="lower", extent=extent, cmap="twilight"
    )
    axes[0, 2].set_title("PHOENIX: phase")
    fig.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].imshow(polarism_frame["rho"], origin="lower", extent=extent)
    axes[1, 0].set_title(f"Polarism: rho (t={polarism_frame['t']:.1f} ps)")
    fig.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(polarism_frame["nR"], origin="lower", extent=extent)
    axes[1, 1].set_title("Polarism: nR")
    fig.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].imshow(
        np.angle(polarism_frame["psi"]), origin="lower", extent=extent, cmap="twilight"
    )
    axes[1, 2].set_title("Polarism: phase")
    fig.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")

    plt.suptitle(f"{case_name}: {frame_name.capitalize()} Frame Comparison")
    _save_fig(outdir, f"frame_{frame_name}_compare.png")


@dataclass
class BenchmarkCase:
    """Store one Phoenix benchmark case."""
    name: str
    final_rel_tol: float
    log_rmse_tol: float
    tail_rel_tol: float
    crossing_dt_tol: float
    tail_fraction: float
    correlation_tol: float = 0.99
    frame_rho_tol: float = 0.10
    frame_nR_tol: float = 0.10
    frame_phase_tol: float = 0.10


BENCHMARK_CASES = [
    BenchmarkCase(
        name="01_uniform_pump",
        final_rel_tol=0.01,
        log_rmse_tol=0.02,
        tail_rel_tol=0.01,
        crossing_dt_tol=0.25,
        tail_fraction=0.5,
        correlation_tol=0.999,
        frame_rho_tol=0.01,
        frame_nR_tol=0.01,
        frame_phase_tol=0.05,
    ),
    BenchmarkCase(
        name="02_pump_no_potential",
        final_rel_tol=0.01,
        log_rmse_tol=0.02,
        tail_rel_tol=0.01,
        crossing_dt_tol=0.25,
        tail_fraction=0.3,
        correlation_tol=0.999,
        frame_rho_tol=0.01,
        frame_nR_tol=0.01,
        frame_phase_tol=0.05,
    ),

    BenchmarkCase(
        name="03_pump_and_potential",
        final_rel_tol=0.05,
        log_rmse_tol=0.05,
        tail_rel_tol=0.15,
        crossing_dt_tol=0.25,
        tail_fraction=0.3,
        correlation_tol=0.999,
        frame_rho_tol=0.05,
        frame_nR_tol=0.05,
        frame_phase_tol=0.25,
    ),
]


def _run_benchmark_case(
    case: BenchmarkCase,
    solver_name: str,
    save_frames: bool = True,
) -> dict:
    """Run benchmark case."""
    base_dir = Path(__file__).resolve().parent / "data" / "phoenix_benchmark"
    case_dir = base_dir / case.name

    rho_max_path = case_dir / "rho_max.txt"
    lasers_yaml = case_dir / "phoenix_lasers_setup.yaml"

    assert rho_max_path.exists(), f"Missing: {rho_max_path}"
    assert lasers_yaml.exists(), f"Missing: {lasers_yaml}"

    validate_benchmark_data(case.name, case_dir, lasers_yaml)

    t_ref, rho_ref = _load_phoenix_rho_max(rho_max_path)

    phoenix_timing = _load_phoenix_timing(case_dir / "timing.json")

    phoenix_frame_first = _load_phoenix_frame(case_dir / "frame_first.npz")
    phoenix_frame_last = _load_phoenix_frame(case_dir / "frame_last.npz")

    cfg = Config()
    configure_simulation(
        cfg=cfg,
        lasers_yaml=lasers_yaml,
        solver_name=solver_name,
        total_time=float(t_ref[-1]),
        use_gpu=compute_engine.use_gpu,
    )
    apply_potential_config(cfg, case.name)

    psi_init_path = case_dir / "psi_init.txt"

    t_sim, rho_sim, elapsed, peak_mb, frames = run_benchmark_simulation(
        cfg, t_ref, psi_init_path=psi_init_path, save_frames=save_frames,
        case_dir=case_dir, case_name=case.name,
    )

    accuracy = compute_accuracy_metrics(
        t_ref, rho_ref, t_sim, rho_sim, tail_fraction=case.tail_fraction
    )

    n_steps = int(t_ref[-1] / cfg.solver.dt)
    steps_per_sec = n_steps / elapsed if elapsed > 0 else 0

    metrics = {
        **accuracy,
        "elapsed_time_s": elapsed,
        "peak_memory_mb": peak_mb,
        "steps_per_second": steps_per_sec,
        "sim_time_ps": float(t_ref[-1]),
    }

    device = "gpu" if compute_engine.use_gpu else "cpu"
    outdir = (
        Path(__file__).resolve().parent
        / "test_results"
        / "test_phoenix_benchmark"
        / f"{case.name}_{solver_name}_{device}"
    )
    plot_benchmark_results(
        outdir=outdir,
        case_name=case.name,
        solver_name=solver_name,
        t_ref=t_ref,
        rho_ref=rho_ref,
        t_sim=t_sim,
        rho_sim=rho_sim,
        metrics=metrics,
        phoenix_timing=phoenix_timing,
    )

    if save_frames:
        plot_frame_comparison(
            outdir=outdir,
            case_name=case.name,
            frame_name="first",
            phoenix_frame=phoenix_frame_first,
            polarism_frame=frames["first"],
        )
        plot_frame_comparison(
            outdir=outdir,
            case_name=case.name,
            frame_name="last",
            phoenix_frame=phoenix_frame_last,
            polarism_frame=frames["last"],
        )

        if phoenix_frame_last is not None and frames["last"] is not None:
            rho_ph = phoenix_frame_last["rho"].astype(np.float64)
            rho_pol = frames["last"]["rho"].astype(np.float64)
            norm = np.sqrt(np.sum(rho_ph**2))
            if norm > 0:
                metrics["frame_rho_rel_l2"] = float(
                    np.sqrt(np.sum((rho_pol - rho_ph) ** 2)) / norm
                )
            else:
                metrics["frame_rho_rel_l2"] = 0.0

            nR_ph = phoenix_frame_last["nR"].astype(np.float64)
            nR_pol = frames["last"]["nR"].astype(np.float64)
            nR_norm = np.sqrt(np.sum(nR_ph**2))
            nR_pol_norm = np.sqrt(np.sum(nR_pol**2))
            if nR_norm > 0:
                metrics["frame_nR_rel_l2"] = float(
                    np.sqrt(np.sum((nR_pol - nR_ph) ** 2)) / nR_norm
                )
            elif nR_pol_norm > 0:
                metrics["frame_nR_rel_l2"] = float("inf")
            else:
                metrics["frame_nR_rel_l2"] = 0.0

            psi_ph = phoenix_frame_last["psi"].astype(np.complex128)
            psi_pol = frames["last"]["psi"].astype(np.complex128)
            rho_threshold = 0.01 * rho_ph.max()
            mask = rho_ph > rho_threshold
            if mask.sum() > 0:
                phase_ph = np.angle(psi_ph[mask])
                phase_pol = np.angle(psi_pol[mask])
                dphase = np.angle(np.exp(1j * (phase_pol - phase_ph)))
                metrics["frame_phase_rmse"] = float(np.sqrt(np.mean(dphase**2)))

    return metrics


FDM_SOLVERS = ["rk4-fdm", "rk4-fdm-fused", "rk4-cuda"]

SOLVER_CASE_MATRIX = [
    (solver, case)
    for solver in FDM_SOLVERS
    for case in BENCHMARK_CASES
]


def _assert_metrics(metrics: dict, case: BenchmarkCase) -> None:
    """Check metrics against the case tolerances."""
    assert metrics["correlation"] >= case.correlation_tol, (
        f"Correlation {metrics['correlation']:.6f} "
        f"below tolerance {case.correlation_tol}"
    )
    assert metrics["log_rmse"] <= case.log_rmse_tol, (
        f"Log RMSE {metrics['log_rmse']:.4f} exceeds tolerance {case.log_rmse_tol}"
    )
    assert metrics["crossing_dt_max"] <= case.crossing_dt_tol, (
        f"Crossing time error {metrics['crossing_dt_max']:.2f} ps "
        f"exceeds tolerance {case.crossing_dt_tol} ps"
    )
    assert metrics["final_rel_error"] <= case.final_rel_tol, (
        f"Final relative error {metrics['final_rel_error']:.4f} "
        f"exceeds tolerance {case.final_rel_tol}"
    )
    if np.isfinite(metrics["max_tail_rel_error"]):
        assert metrics["max_tail_rel_error"] <= case.tail_rel_tol, (
            f"Tail relative error {metrics['max_tail_rel_error']:.4f} "
            f"exceeds tolerance {case.tail_rel_tol}"
        )
    if "frame_rho_rel_l2" in metrics:
        assert metrics["frame_rho_rel_l2"] <= case.frame_rho_tol, (
            f"Last-frame density L2 error {metrics['frame_rho_rel_l2']:.4f} "
            f"exceeds tolerance {case.frame_rho_tol}"
        )
    if "frame_nR_rel_l2" in metrics:
        assert metrics["frame_nR_rel_l2"] <= case.frame_nR_tol, (
            f"Last-frame reservoir nR L2 error {metrics['frame_nR_rel_l2']:.4f} "
            f"exceeds tolerance {case.frame_nR_tol}"
        )
    if "frame_phase_rmse" in metrics:
        assert metrics["frame_phase_rmse"] <= case.frame_phase_tol, (
            f"Last-frame phase RMSE {metrics['frame_phase_rmse']:.4f} rad "
            f"exceeds tolerance {case.frame_phase_tol}"
        )


@pytest.mark.parametrize(
    "solver_name,case",
    SOLVER_CASE_MATRIX,
    ids=[f"{s}-{c.name}" for s, c in SOLVER_CASE_MATRIX],
)
def test_phoenix_accuracy(solver_name: str, case: BenchmarkCase):
    """Test that phoenix accuracy."""
    metrics = _run_benchmark_case(case, solver_name)
    _assert_metrics(metrics, case)
