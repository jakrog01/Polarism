from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from polarism.simulation_grid_2D import SimulationGrid2D


@dataclass(frozen=True)
class GridCfg:
    nx: int = 100
    ny: int = 100
    lx: float = 200.0
    ly: float = 200.0


class NoBoundaryCondition:
    def before_step_action(self):
        return 0.0

    def after_step_action(self, psi):
        return psi


class State:
    def __init__(self, psi: np.ndarray):
        self.psi = psi
        self.t = 0.0


class NullReservoir:
    def __init__(self, grid: SimulationGrid2D):
        self._n0 = np.zeros((grid.nx, grid.ny), dtype=np.float64)
        self._state = (self._n0,)

    def step(self, dt: float, psi, Pxy) -> None:
        return None

    def get_reservoir_density(self):
        return self._n0

    def get_reservoir_densities(self):
        return self._n0, self._n0

    def make_result_nodes(self):
        return []

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def get_derivatives(self, psi, pump, state):
        return tuple(np.zeros_like(s) for s in state)

    def get_active_density(self, state):
        return state[0]


def make_grid(cfg: GridCfg | None = None) -> SimulationGrid2D:
    return SimulationGrid2D(cfg or GridCfg())


def make_reservoir_cfg(expose_results: bool = False):
    return SimpleNamespace(expose_results=expose_results)


def make_physics_default(**overrides):
    base = dict(
        hbar=0.6582119514,
        m_eff=0.284,
        gamma_R=0.005,
        gamma_C=0.083,
        g_C=0.001,
        g_R=0.002,
        R=0.02,
        gamma_I=0.001,
        gamma_A=0.005,
        R_IA=5e-2,
        R_AI=2.5e-3,
        D=0.0,
        D_I=0.0,
        D_A=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_physics_linear(**overrides):
    base = dict(
        hbar=0.6582119514,
        m_eff=0.284,
        gamma_R=0.0,
        gamma_C=0.0,
        g_C=0.0,
        g_R=0.0,
        R=0.0,
        gamma_I=0.0,
        gamma_A=0.0,
        R_IA=0.0,
        R_AI=0.0,
        D=0.0,
        D_I=0.0,
        D_A=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_config(dt: float, physics):
    return SimpleNamespace(solver=SimpleNamespace(dt=float(dt)), physics=physics)


def pump_uniform(P0: float, X: np.ndarray) -> np.ndarray:
    return np.ones_like(X, dtype=np.float64) * float(P0)


def pump_zero(X: np.ndarray) -> np.ndarray:
    return np.zeros_like(X, dtype=np.float64)


def potential_zero(X: np.ndarray) -> np.ndarray:
    return np.zeros_like(X, dtype=np.float64)


def potential_periodic(grid: SimulationGrid2D, V0: float = 0.7) -> np.ndarray:
    kx = 2 * np.pi / grid.lx
    ky = 2 * np.pi / grid.ly
    return (V0 * (np.cos(kx * grid.X) + np.cos(ky * grid.Y))).astype(np.float64)


def initial_uniform(
    grid: SimulationGrid2D, amp: float, phase: float = 0.0
) -> np.ndarray:
    return (amp * np.exp(1j * phase)) * np.ones((grid.nx, grid.ny), dtype=np.complex128)


def initial_packet(
    grid: SimulationGrid2D, sigma: float = 25.0, kx: float = 0.15
) -> np.ndarray:
    env = np.exp(-0.5 * (grid.X**2 + grid.Y**2) / (sigma**2))
    phase = np.exp(1j * kx * grid.X)
    psi0 = (env * phase).astype(np.complex128)
    nrm = np.linalg.norm(psi0.ravel())
    return psi0 / (nrm + 1e-30)


def max_abs(a: np.ndarray) -> float:
    return float(np.max(np.abs(a)))


def uniformity_error(field: np.ndarray) -> float:
    return max_abs(field - np.mean(field))


def grad_central_roll(f: np.ndarray, d: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * d)


def laplacian_roll(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / (dx * dx) + (
        np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
    ) / (dy * dy)


def align_global_phase(psi: np.ndarray, ref: np.ndarray) -> np.ndarray:
    inner = np.vdot(ref.ravel(), psi.ravel())
    if inner == 0:
        return psi
    return psi * np.exp(-1j * np.angle(inner))


def rel_l2_err(psi: np.ndarray, ref: np.ndarray) -> float:
    psi = align_global_phase(psi, ref)
    num = np.linalg.norm((psi - ref).ravel())
    den = np.linalg.norm(ref.ravel())
    return float(num / (den + 1e-30))


class Recorder:
    def __init__(self):
        self.t = []
        self.metrics = {}

    def add(self, t: float, **kwargs):
        self.t.append(float(t))
        for k, v in kwargs.items():
            self.metrics.setdefault(k, []).append(float(v))

    def plot(self, path: Path, title: str, y_keys: list[str], y_log: bool = False):
        plt.figure()
        for k in y_keys:
            if k in self.metrics:
                plt.plot(self.t, self.metrics[k], label=k)
        plt.xlabel("t")
        plt.ylabel("value")
        if y_log:
            plt.yscale("log")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
