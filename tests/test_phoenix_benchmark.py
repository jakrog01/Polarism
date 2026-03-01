import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController


def _apply_phoenix_benchmark_cfg(cfg: Config, lasers_yaml: str) -> None:
    cfg.grid.nx = 128
    cfg.grid.ny = 128
    cfg.grid.lx = 20.0
    cfg.grid.ly = 20.0

    cfg.boundary_condition.absorption = "no-absorption"
    cfg.boundary_condition.strength = 0.0

    cfg.reservoir.reservoir_type = "single"

    cfg.physics.hbar = 6.582119569e-4
    cfg.physics.m_eff = 0.5 * 5.864e-4
    cfg.physics.gamma_C = 0.16
    cfg.physics.gamma_R = 0.24
    cfg.physics.R = 0.01
    cfg.physics.g_C = 6e-6
    cfg.physics.g_R = 1.2e-5
    cfg.physics.D = 0.0

    cfg.solver.method = "split-step-fft"
    cfg.solver.dt = 0.02
    cfg.solver.total_time = 2500.0

    cfg.potential.potential_type = "double-well-supergaussian"
    cfg.potential.x1, cfg.potential.y1 = -2.0, 0.0
    cfg.potential.x2, cfg.potential.y2 = 2.0, 0.0
    cfg.potential.V1, cfg.potential.V2 = -0.002, -0.0022
    cfg.potential.w1, cfg.potential.w2 = 1.5, 1.5
    cfg.potential.order = 2.0

    cfg.laser.mode = "multiple"
    cfg.laser.config_file = lasers_yaml

    cfg.result.real_time_view = False
    cfg.result.save_results = False
    cfg.result.save_hdf5 = False
    cfg.result.save_json = False
    cfg.result.save_npy = False


def _asnumpy(x):
    xp = compute_engine.xp
    return xp.asnumpy(x) if hasattr(xp, "asnumpy") else np.asarray(x)


def _rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))


def _load_wavefunction_plus(path: Path, nx: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.shape != (2 * nx, nx):
        raise AssertionError(
            f"Expected wavefunction shape {(2*nx, nx)}, got {data.shape}"
        )
    re_part = data[:nx, :]
    im_part = data[nx:, :]
    return re_part + 1j * im_part


def _save_fig(outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / name, dpi=160)
    plt.close()


def test_phoenix_result_integrity():
    base = Path(__file__).resolve().parent / "data" / "phoenix_benchmark"
    pump_path = base / "pump_plus.txt"
    pot_path = base / "potential_plus.txt"
    wave_path = base / "wavefunction_plus.txt"
    lasers_yaml = base / "phoenix_lasers_setup.yaml"
    rho_max_path = base / "rho_max_plus.txt"

    missing = [
        p for p in [pump_path, pot_path, wave_path, lasers_yaml] if not p.exists()
    ]
    if missing:
        pytest.skip("Missing benchmark assets: " + ", ".join(str(p) for p in missing))

    cfg = Config()
    _apply_phoenix_benchmark_cfg(cfg, lasers_yaml=str(lasers_yaml))

    pump_ref = np.loadtxt(pump_path, comments="#")
    pot_ref = np.loadtxt(pot_path, comments="#")
    psi_ref = _load_wavefunction_plus(wave_path, nx=cfg.grid.nx)

    pho = np.loadtxt(rho_max_path, comments="#")
    t_ph = pho[:, 0].astype(float)
    rm_ph = pho[:, 1].astype(float)
    idx = np.argsort(t_ph)
    t_ph = t_ph[idx]
    rm_ph = rm_ph[idx]
    t_unique, inv = np.unique(t_ph, return_inverse=True)
    rm_unique = np.empty_like(t_unique)
    rm_unique[:] = np.nan
    rm_unique[inv] = rm_ph

    t_ph = t_unique
    rm_ph = rm_unique

    sim = SimulationController(cfg)
    xp = compute_engine.xp

    P0 = sim._compute_total_pump(0.0)
    pump_sim = _asnumpy(P0)

    pot_sim_arr = _asnumpy(sim.potential)
    pot_sim = np.real(pot_sim_arr)

    err_pump = _rel_l2(pump_sim, pump_ref)
    err_pot = _rel_l2(pot_sim, pot_ref)

    assert err_pump < 5e-3, f"Pump mismatch vs PHOENIX (rel-L2={err_pump:.3e})"
    assert err_pot < 5e-3, f"Potential mismatch vs PHOENIX (rel-L2={err_pot:.3e})"

    sim.reservoir.set_state((P0,))
    sim.state.psi = xp.asarray((1e-7 * _asnumpy(P0)).astype(np.complex128))

    dt = cfg.solver.dt
    n_steps = int(cfg.solver.total_time / dt)
    out_every = 0.5
    stride = max(1, int(round(out_every / dt)))

    maskL = ((sim.grid.X + 2.0) ** 2 + (sim.grid.Y - 0.0) ** 2) <= (2.0**2)
    maskR = ((sim.grid.X - 2.0) ** 2 + (sim.grid.Y - 0.0) ** 2) <= (2.0**2)

    times, Ntot, NL, NR, Nmax = [], [], [], [], []
    for step in range(n_steps):
        sim.state.t = step * dt
        sim.solver.step(
            sim.potential, P0, sim.reservoir, sim.boundary_condition, sim.state
        )

        if step % stride == 0:
            rho = xp.abs(sim.state.psi) ** 2
            times.append(sim.state.t)
            Ntot.append(float(rho.sum()))
            NL.append(float((rho * maskL).sum()))
            NR.append(float((rho * maskR).sum()))
            Nmax.append(float(rho.max()))

    psi_sim = _asnumpy(sim.state.psi)

    rho_sim = np.abs(psi_sim) ** 2
    rho_ref = np.abs(psi_ref) ** 2

    rho_max_sim_final = float(rho_sim.max())
    rho_max_phoenix_final = float(rho_ref.max())

    rho_sim_n = rho_sim / (rho_sim.sum() + 1e-30)
    rho_ref_n = rho_ref / (rho_ref.sum() + 1e-30)

    assert not np.shares_memory(pump_sim, pump_ref)
    assert not np.shares_memory(psi_sim, psi_ref)

    err_rho = _rel_l2(rho_sim_n, rho_ref_n)
    assert err_rho < 0.10, f"Final density mismatch vs PHOENIX (rel-L2={err_rho:.3f})"

    outdir = Path(__file__).resolve().parent / "test_results" / "test_phoenix_benchmark"

    for name, a, b in [
        ("pump", pump_sim, pump_ref),
        ("potential", pot_sim, pot_ref),
        ("rho", rho_sim_n, rho_ref_n),
    ]:
        plt.figure(figsize=(12, 3.2))
        plt.subplot(1, 3, 1)
        plt.title(f"{name}: sim")
        plt.imshow(a, origin="lower")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 3, 2)
        plt.title(f"{name}: phoenix")
        plt.imshow(b, origin="lower")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1, 3, 3)
        plt.title(f"{name}: diff")
        plt.imshow(a - b, origin="lower")
        plt.colorbar(fraction=0.046, pad=0.04)
        _save_fig(outdir, f"{name}_compare.png")

    mid = rho_sim_n.shape[1] // 2
    plt.figure(figsize=(9, 3))
    plt.title("Center cut (y mid): rho normalized")
    plt.plot(rho_sim_n[:, mid], label="sim")
    plt.plot(rho_ref_n[:, mid], "--", label="phoenix")
    plt.legend()
    _save_fig(outdir, "rho_center_cut.png")

    plt.figure(figsize=(9, 3))
    plt.title("Time traces (sim)")
    plt.plot(times, Ntot, label="Ntot")
    plt.plot(times, NL, label="NL")
    plt.plot(times, NR, label="NR")
    plt.legend()
    _save_fig(outdir, "time_traces.png")

    plt.figure(figsize=(9, 3))
    plt.title("Max density vs time (sim vs phoenix)")
    plt.plot(times, Nmax, label="max rho (sim)")
    plt.plot(t_ph, rm_ph, "--", label="max rho (phoenix)")

    plt.scatter([times[-1]], [Nmax[-1]], marker="o", label="sim last")
    plt.scatter([t_ph[-1]], [rm_ph[-1]], marker="x", label="phoenix last")

    x0 = max(min(times), float(t_ph.min()))
    x1 = min(max(times), float(t_ph.max()))
    if x0 < x1:
        plt.xlim(x0, x1)

    plt.xlabel("t (ps)")
    plt.ylabel("max |psi|^2")
    plt.legend()
    _save_fig(outdir, "max_density_compare.png")

    with open(outdir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"err_pump_relL2 = {err_pump:.6e}\n")
        f.write(f"err_potential_relL2 = {err_pot:.6e}\n")
        f.write(f"err_rho_relL2(normalized) = {err_rho:.6e}\n")
        f.write(f"rho_max_sim_final = {rho_max_sim_final:.6e}\n")
        f.write(f"rho_max_phoenix_final = {rho_max_phoenix_final:.6e}\n")
