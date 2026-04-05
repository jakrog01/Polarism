"""Physics simulation kernel for the GPE and reservoir model.

This module only handles the simulation work. It does not know about
Slurm, config-file layout, or pipeline directories.
"""
from __future__ import annotations

import math
import os
import traceback

import h5py
import numpy as np

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.grid.create_grid import create_grid
from polarism.potential.create_potential import create_potential
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState
from polarism.solver.create_solver import create_solver
from tqdm import trange

RECORD_STRIDE = 100
COND_GROWTH_FACTOR = 1e6
CHECK_EVERY = 50
MIN_CHECK_TIME = 50.0
RESERVED_GPU_GB = 4.0
RNG_SEED = 42

class AsyncBatchWriter:
    """Write HDF5 frames with double buffering.

    On GPU runs it uses pinned host buffers and a non-blocking stream when
    that path is available. Otherwise it falls back to a normal copy.
    """

    def __init__(self, filepath: str, batch_size: int, ny: int, nx: int) -> None:
        """Set up batch buffers and the HDF5 file."""
        self.xp = compute_engine.xp
        self.use_gpu = compute_engine.use_gpu
        self.batch_size = batch_size
        self.ny = ny
        self.nx = nx
        self.h5 = h5py.File(filepath, "w")
        self.datasets: dict = {}
        self.total = 0

        self._cnames = ["psi"]
        self._rnames = ["nI", "nA", "Pump"]
        self._all_names = self._cnames + self._rnames

        self._gpu = [{}, {}]
        for i in range(2):
            for nm in self._cnames:
                self._gpu[i][nm] = self.xp.zeros(
                    (batch_size, ny, nx), dtype=self.xp.complex128
                )
            for nm in self._rnames:
                self._gpu[i][nm] = self.xp.zeros(
                    (batch_size, ny, nx), dtype=self.xp.float64
                )

        self._host: dict = {}
        self._pinned = False
        if self.use_gpu:
            try:
                import cupy as cp

                for nm in self._cnames:
                    buf = np.empty((batch_size, ny, nx), dtype=np.complex128)
                    cp.cuda.runtime.hostRegister(buf.ctypes.data, buf.nbytes, 0)
                    self._host[nm] = buf
                for nm in self._rnames:
                    buf = np.empty((batch_size, ny, nx), dtype=np.float64)
                    cp.cuda.runtime.hostRegister(buf.ctypes.data, buf.nbytes, 0)
                    self._host[nm] = buf
                self._transfer_stream = cp.cuda.Stream(non_blocking=True)
                self._event = cp.cuda.Event()
                self._pinned = True
            except Exception:
                self._host = {}
                self._pinned = False

        if not self._pinned:
            for nm in self._cnames:
                self._host[nm] = np.empty((batch_size, ny, nx), dtype=np.complex128)
            for nm in self._rnames:
                self._host[nm] = np.empty((batch_size, ny, nx), dtype=np.float64)

        self._active = 0
        self._count = 0
        self._transfer_pending = False
        self._pending_n = 0
        self._time_buf: list = []
        self._mode_buf: list = []
        self._scalar_bufs: dict = {}
        self._pend_time = None
        self._pend_mode = None
        self._pend_scalars = None
        self._writes_since_flush = 0
        self._flush_every = 10

    def register_scalar(self, name: str) -> None:
        """Register one scalar series for output."""
        self._scalar_bufs[name] = []

    def record(self, t: float, fields: dict, scalars: dict, mode: int = 0) -> None:
        """Record one frame in the active batch."""
        if self._transfer_pending:
            self._wait_and_write()

        idx = self._count
        buf = self._gpu[self._active]

        self._time_buf.append(t)
        self._mode_buf.append(mode)
        for nm, data in fields.items():
            buf[nm][idx] = data
        for nm, val in scalars.items():
            self._scalar_bufs[nm].append(float(val))

        self._count += 1
        if self._count >= self.batch_size:
            self._start_transfer()

    def _start_transfer(self) -> None:
        """Start copying the active batch to host memory."""
        n = self._count
        buf = self._gpu[self._active]

        if self._pinned:
            for nm in self._all_names:
                buf[nm][:n].get(out=self._host[nm][:n], stream=self._transfer_stream)
            self._event.record(self._transfer_stream)
        else:
            for nm in self._all_names:
                self._host[nm][:n] = compute_engine.to_cpu(buf[nm][:n])

        self._pend_time = list(self._time_buf)
        self._pend_mode = list(self._mode_buf)
        self._pend_scalars = {k: list(v) for k, v in self._scalar_bufs.items()}
        self._pending_n = n
        self._transfer_pending = True

        self._active = 1 - self._active
        self._count = 0
        self._time_buf = []
        self._mode_buf = []
        for k in self._scalar_bufs:
            self._scalar_bufs[k] = []

    def _wait_and_write(self) -> None:
        """Wait for a pending copy and write it to disk."""
        if self._pinned:
            self._event.synchronize()

        n = self._pending_n
        self._append("time", np.array(self._pend_time[:n]), scalar=True)
        self._append("mode", np.array(self._pend_mode[:n], dtype=np.int8), scalar=True)

        for nm in self._all_names:
            self._append(f"fields/{nm}", self._host[nm][:n], scalar=False)

        for nm, vals in self._pend_scalars.items():
            self._append(f"scalars/{nm}", np.array(vals[:n]), scalar=True)

        self.total += n
        self._transfer_pending = False
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self.h5.flush()
            self._writes_since_flush = 0

    def _append(self, name: str, data: np.ndarray, scalar: bool) -> None:
        """Append data to one HDF5 dataset."""
        if name not in self.datasets:
            if scalar:
                self.datasets[name] = self.h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None,),
                    compression="lzf",
                    chunks=(min(1000, self.batch_size),),
                )
            else:
                self.datasets[name] = self.h5.create_dataset(
                    name,
                    data=data,
                    maxshape=(None, *data.shape[1:]),
                    compression="lzf",
                    chunks=(min(32, self.batch_size), *data.shape[1:]),
                )
        else:
            ds = self.datasets[name]
            old = ds.shape[0]
            ds.resize(old + data.shape[0], axis=0)
            ds[old:] = data

    def close(self) -> None:
        """Flush pending data and close the writer."""
        if self._transfer_pending:
            self._wait_and_write()
        if self._count > 0:
            self._start_transfer()
            self._wait_and_write()
        if self._pinned:
            import cupy as cp

            for buf in self._host.values():
                try:
                    cp.cuda.runtime.hostUnregister(buf.ctypes.data)
                except Exception:
                    pass
        self.h5.close()


def check_condensation_kspace(psi: object, xp: object) -> float:
    """Return the k=0 power fraction of *psi* (FFT-based)."""
    psi_k = xp.fft.fft2(psi)
    power = xp.abs(psi_k) ** 2
    k0_power = float(power[0, 0])
    total_power = float(xp.sum(power))
    if total_power < 1e-30:
        return 0.0
    return k0_power / total_power


def compute_batch_size(ny: int, nx: int) -> int:
    """Estimate a safe HDF5 write-buffer depth for the current GPU.

    Allocates 75 % of free GPU memory (after reserving ``RESERVED_GPU_GB``
    GB) across both double-buffer GPU slots.  Returns 1 when memory is too
    tight to fit even a single frame per slot — correct but slower, as every
    ``record()`` call triggers an immediate transfer.  Returns 500 on the
    CPU path where host memory is not the limiting resource.

    Parameters
    ----------
    ny, nx : int
        Spatial grid dimensions.

    Returns
    -------
    int
        Batch depth in frames, in [1, 10000].
    """
    xp = compute_engine.xp
    if not hasattr(xp, "cuda"):
        return 500
    free = xp.cuda.Device().mem_info[0]
    usable = max(0, free - int(RESERVED_GPU_GB * 1024**3))
    frame_bytes = ny * nx * (16 + 8 + 8 + 8)
    batch = int(usable * 0.75) // (2 * frame_bytes) if frame_bytes else 0
    return max(1, min(batch, 10000))


def _precompute_spatial_profiles(lasers: list, grid: object) -> list:
    """Precompute the spatial pump profile for each laser."""
    return [laser._P_space(grid.X, grid.Y) for laser in lasers]


def _p_time_pure(
    t: float,
    pulse_separation: float,
    cutoff_sigma: float,
    sigma_time: float,
) -> float:
    """Return the pulse envelope at time t."""
    phase = cutoff_sigma * sigma_time
    n = max(0, round((t - phase) / pulse_separation))
    dt_val = t - n * pulse_separation - phase
    if abs(dt_val) > cutoff_sigma * sigma_time:
        return 0.0
    return math.exp(-0.5 * (dt_val / sigma_time) ** 2)


def _precompute_spatial_maxes(profiles: list, xp: object) -> list:
    """Precompute the peak value of each spatial profile."""
    return [float(xp.max(p)) for p in profiles]


def _compute_pump_fast(
    lasers: list,
    profiles: list,
    spatial_maxes: list,
    t: float,
    xp: object,
    P_zero: object,
) -> tuple:
    """Build the total pump field and per-laser peaks."""
    P_total = P_zero.copy()
    per_laser_max = []
    for i, laser in enumerate(lasers):
        t_eff = t - laser.delay
        if t_eff < 0:
            per_laser_max.append(0.0)
            continue
        amp = laser._amplitude(t_eff)
        pt = _p_time_pure(t_eff, laser.pulse_separation, laser.cutoff_sigma, laser.sigma_time)
        temporal = float(amp) * pt
        if temporal == 0.0:
            per_laser_max.append(0.0)
            continue
        P_total += temporal * profiles[i]
        per_laser_max.append(temporal * spatial_maxes[i])
    return P_total, per_laser_max


def run_simulation_from_config(
    routine_name: str,
    lasers: list,
    cfg: Config,
    batch_size: int,
    output_dir: str,
) -> float | None:
    """Run a simulation with an explicit ``Config`` object.

    Parameters
    ----------
    routine_name : str
        Output filename stem. HDF5 is written as
        ``<output_dir>/<routine_name>.h5``.
    lasers : list
        Laser instances (e.g. ``PulseGaussian``).
    cfg : Config
        Fully constructed simulation config from :mod:`pipeline.config.builder`.
    batch_size : int
        HDF5 write-buffer depth in frames; use :func:`compute_batch_size`.
    output_dir : str
        Directory for the output HDF5 file.  Must already exist.

    Returns
    -------
    float or None
        Condensation time in ps, or ``None`` if no condensation detected.

    Raises
    ------
    Exception
        Any simulation-loop exception is logged and re-raised so the calling
        Slurm job exits nonzero.
    """
    xp = compute_engine.xp

    grid = create_grid(cfg.grid)
    bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics)
    potential = create_potential(cfg.potential, grid)
    reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid)

    rng_gpu = xp.random.default_rng(RNG_SEED)
    state = SimulationState.__new__(SimulationState)
    state.psi = (
        cfg.physics.init_eps
        * (
            rng_gpu.random((grid.ny, grid.nx), dtype=xp.float64)
            + 1j * rng_gpu.random((grid.ny, grid.nx), dtype=xp.float64)
        )
    ).astype(xp.complex128)
    state.t = 0.0

    solver = create_solver(cfg, grid)

    cap = bc.before_step_action()
    if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
        potential = potential.astype(state.psi.dtype)
        cap = cap.astype(state.psi.dtype)
    potential = potential + cap

    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    dt = cfg.solver.dt
    print(f"    n_steps={n_steps:,}, dt={dt}")

    out_path = os.path.join(output_dir, f"{routine_name}.h5")
    writer = AsyncBatchWriter(out_path, batch_size, grid.ny, grid.nx)

    scalar_names = ["psi_sq_max", "nI_max", "nA_max", "k0_frac", "P_max"]
    for li in range(len(lasers)):
        scalar_names.append(f"P_max_{li}")
    for name in scalar_names:
        writer.register_scalar(name)

    spatial_profiles = _precompute_spatial_profiles(lasers, grid)
    spatial_maxes = _precompute_spatial_maxes(spatial_profiles, xp)
    P_zero = xp.zeros((grid.ny, grid.nx), dtype=xp.float64)

    N_initial = float(xp.sum(xp.abs(state.psi) ** 2))
    t_cond = None
    condensed = False
    last_t = 0.0
    step = 0

    try:
        for step in trange(n_steps, desc=f"  {routine_name}"):
            t = step * dt
            last_t = t

            P_total, per_laser_max = _compute_pump_fast(
                lasers, spatial_profiles, spatial_maxes, t, xp, P_zero
            )
            solver.step(potential, P_total, reservoir, bc, state)

            if step > 0 and step % CHECK_EVERY == 0:
                N_total = float(xp.sum(xp.abs(state.psi) ** 2))
                if math.isnan(N_total) or math.isinf(N_total):
                    raise RuntimeError(
                        f"Numerical divergence at step {step}, t={t:.4f} ps: "
                        f"psi norm is {N_total}"
                    )
                if not condensed and t >= MIN_CHECK_TIME and N_total > N_initial * COND_GROWTH_FACTOR:
                    t_cond = (step + 1) * dt
                    condensed = True
                    print(f"\n    Condensation at t={t:.2f} ps (step {step})")

            if step % RECORD_STRIDE == 0:
                nA, nI = reservoir.get_state()
                psi_sq = xp.abs(state.psi) ** 2
                k0_f = (
                    check_condensation_kspace(state.psi, xp)
                    if step % CHECK_EVERY == 0
                    else 0.0
                )
                scalars = {
                    "psi_sq_max": float(xp.max(psi_sq)),
                    "nI_max": float(xp.max(nI)),
                    "nA_max": float(xp.max(nA)),
                    "k0_frac": k0_f,
                    "P_max": float(xp.max(P_total)),
                }
                for li, lp in enumerate(per_laser_max):
                    scalars[f"P_max_{li}"] = lp
                writer.record(
                    (step + 1) * dt,
                    {"psi": state.psi, "nI": nI, "nA": nA, "Pump": P_total},
                    scalars,
                )
    except Exception as e:
        print(f"\n    ERROR at step {step}, t={last_t:.2f} ps: {e}")
        traceback.print_exc()
        raise
    finally:
        print("    Closing HDF5 writer ...")
        writer.close()
        print("    HDF5 finalized.")

    print(
        f"    -> {out_path}  ({writer.total} frames, "
        f"last_t={last_t:.1f} ps, t_cond={t_cond})"
    )
    return t_cond
