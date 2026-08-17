"""Shared test helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import polarism.laser
from polarism.grid.create_grid import create_grid
from polarism.grid.simulation_grid_2d import SimulationGrid2D


@dataclass(frozen=True)
class GridCfg:
    """Small grid config used in tests."""
    nx: int = 100
    ny: int = 100
    lx: float = 200.0
    ly: float = 200.0
    grid_type: str = "periodic"


@dataclass(frozen=True)
class PeakMemoryResult:
    """Peak process and optional CUDA memory measurements."""

    peak_rss_bytes: int | None
    peak_gpu_memory_bytes: int | None
    peak_gpu_memory_reason: str | None


class PeakMemoryMonitor:
    """Sample process RSS and CUDA pool usage during a measured operation."""

    def __init__(self, measure_gpu: bool = False, interval_seconds: float = 0.005):
        self._measure_gpu = measure_gpu
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._peak_rss_bytes: int | None = None
        self._peak_gpu_memory_bytes: int | None = None
        self._peak_gpu_memory_reason: str | None = None

    @staticmethod
    def _read_rss_bytes() -> int | None:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _sample_rss_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_rss_once()
            self._stop.wait(self._interval_seconds)

    def _sample_rss_once(self) -> None:
        value = self._read_rss_bytes()
        if value is not None:
            self._peak_rss_bytes = max(self._peak_rss_bytes or value, value)

    def sample_gpu(self) -> None:
        """Update the CUDA memory high-water measurement when requested."""
        if not self._measure_gpu:
            return
        try:
            import cupy

            pool = cupy.get_default_memory_pool()
            value = max(int(pool.used_bytes()), int(pool.total_bytes()))
            self._peak_gpu_memory_bytes = max(
                self._peak_gpu_memory_bytes or value, value
            )
            self._peak_gpu_memory_reason = None
        except Exception as error:
            self._peak_gpu_memory_reason = f"CUDA memory measurement unavailable: {error}"

    def __enter__(self) -> "PeakMemoryMonitor":
        self._stop.clear()
        self._sample_rss_once()
        self.sample_gpu()
        self._thread = Thread(target=self._sample_rss_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.sample_gpu()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 4.0 * self._interval_seconds))
        self._sample_rss_once()

    def result(self) -> PeakMemoryResult:
        """Return measurements collected after leaving the context."""
        gpu_reason = self._peak_gpu_memory_reason
        if self._measure_gpu and self._peak_gpu_memory_bytes is None and gpu_reason is None:
            gpu_reason = "CUDA memory measurement produced no samples"
        return PeakMemoryResult(
            peak_rss_bytes=self._peak_rss_bytes,
            peak_gpu_memory_bytes=self._peak_gpu_memory_bytes,
            peak_gpu_memory_reason=gpu_reason,
        )


class NoBoundaryCondition:
    """Test helper with no boundary action."""
    def before_step_action(self):
        """Return zero boundary contribution."""
        return 0.0

    def after_step_action(self, psi):
        """Return psi unchanged."""
        return psi


class State:
    """Small test state wrapper."""
    def __init__(self, psi: np.ndarray):
        """Store the test field."""
        self.psi = psi
        self.t = 0.0


class NullReservoir:
    """Test helper reservoir with zero density."""
    def __init__(self, grid: SimulationGrid2D):
        """Set up zero reservoir arrays."""
        self._n0 = np.zeros((grid.ny, grid.nx), dtype=np.float64)
        self._state = (self._n0,)

    def step(self, dt: float, psi, Pxy) -> None:
        """Advance the null reservoir by one step."""
        return None

    def get_reservoir_density(self):
        """Return the total reservoir density."""
        return self._n0

    def get_reservoir_densities(self):
        """Return the active and inactive densities."""
        return self._n0, self._n0

    def make_result_nodes(self):
        """Return no result nodes."""
        return []

    def get_state(self):
        """Return the stored reservoir state."""
        return self._state

    def set_state(self, state):
        """Store the reservoir state."""
        self._state = state

    def get_derivatives(self, psi, pump, state):
        """Return zero reservoir derivatives."""
        return tuple(np.zeros_like(s) for s in state)

    def get_active_density(self, state):
        """Return the active reservoir density."""
        return state[0]


def make_grid(cfg: GridCfg | None = None) -> SimulationGrid2D:
    """Build a test grid."""
    return create_grid(cfg or GridCfg())


def make_reservoir_cfg(expose_results: bool = False):
    """Build a simple reservoir config stub."""
    return SimpleNamespace(expose_results=expose_results)


def make_physics_default(**overrides):
    """Build default physics values for tests."""
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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_physics_linear(**overrides):
    """Build linear physics values for tests."""
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
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def make_config(
    dt: float,
    physics,
    grid_type: str = "periodic",
    reservoir_type: str = "single",
):
    """Build a small config stub for tests."""
    return SimpleNamespace(
        solver=SimpleNamespace(dt=float(dt)),
        physics=physics,
        grid=SimpleNamespace(grid_type=grid_type),
        reservoir=SimpleNamespace(reservoir_type=reservoir_type),
    )


def pump_uniform(P0: float, X: np.ndarray) -> np.ndarray:
    """Return a uniform pump field."""
    return np.ones_like(X, dtype=np.float64) * float(P0)


def pump_zero(X: np.ndarray) -> np.ndarray:
    """Return a zero pump field."""
    return np.zeros_like(X, dtype=np.float64)


def potential_zero(X: np.ndarray) -> np.ndarray:
    """Return a zero potential field."""
    return np.zeros_like(X, dtype=np.float64)


def potential_periodic(grid: SimulationGrid2D, V0: float = 0.7) -> np.ndarray:
    """Return a cosine lattice potential."""
    kx = 2 * np.pi / grid.lx
    ky = 2 * np.pi / grid.ly
    return (V0 * (np.cos(kx * grid.X) + np.cos(ky * grid.Y))).astype(np.float64)


def initial_uniform(
    grid: SimulationGrid2D, amp: float, phase: float = 0.0
) -> np.ndarray:
    """Return a uniform initial field."""
    return (amp * np.exp(1j * phase)) * np.ones((grid.ny, grid.nx), dtype=np.complex128)


def initial_packet(
    grid: SimulationGrid2D, sigma: float = 25.0, kx: float = 0.15
) -> np.ndarray:
    """Return a normalized Gaussian wave packet."""
    env = np.exp(-0.5 * (grid.X**2 + grid.Y**2) / (sigma**2))
    phase = np.exp(1j * kx * grid.X)
    psi0 = (env * phase).astype(np.complex128)
    nrm = np.linalg.norm(psi0.ravel())
    return psi0 / (nrm + 1e-30)


def max_abs(a: np.ndarray) -> float:
    """Return the largest absolute value."""
    return float(np.max(np.abs(a)))


def uniformity_error(field: np.ndarray) -> float:
    """Return the max deviation from the mean."""
    return max_abs(field - np.mean(field))


def grad_central_roll(f: np.ndarray, d: float, axis: int) -> np.ndarray:
    """Compute a central-difference gradient with rolls."""
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * d)


def laplacian_roll(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute the periodic Laplacian with rolls."""
    return (np.roll(f, -1, axis=0) - 2 * f + np.roll(f, 1, axis=0)) / (dx * dx) + (
        np.roll(f, -1, axis=1) - 2 * f + np.roll(f, 1, axis=1)
    ) / (dy * dy)


def align_global_phase(psi: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Align psi to the phase of ref."""
    inner = np.vdot(ref.ravel(), psi.ravel())
    if inner == 0:
        return psi
    return psi * np.exp(-1j * np.angle(inner))


def rel_l2_err(psi: np.ndarray, ref: np.ndarray) -> float:
    """Return the relative L2 error after phase alignment."""
    psi = align_global_phase(psi, ref)
    num = np.linalg.norm((psi - ref).ravel())
    den = np.linalg.norm(ref.ravel())
    return float(num / (den + 1e-30))


class Recorder:
    """Collect scalar series for test plots."""
    def __init__(self):
        """Set up empty recorded series."""
        self.t = []
        self.metrics = {}

    def add(self, t: float, **kwargs):
        """Record one scalar step."""
        self.t.append(float(t))
        for k, v in kwargs.items():
            self.metrics.setdefault(k, []).append(float(v))

    def plot(self, path: Path, title: str, y_keys: list[str], y_log: bool = False):
        """Plot selected recorded series."""
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
