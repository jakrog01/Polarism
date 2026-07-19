"""Batched GPU simulation runner for the dynamic SNN pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.init_condition import make_initial_psi
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState

from snn_dynamic.simulation.resources import SharedSimResources
from snn_dynamic.simulation.runner import _output_steps, _pulse_time_factors

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchedTraces:
    """Dynamic readout traces for encoded image batches.

    Attributes
    ----------
    psi
        Condensate density traces with shape ``(N, T_out, 50)``.
    n_active
        Active reservoir traces with shape ``(N, T_out, 50)``, or ``None``.
    n_inactive
        Inactive reservoir traces with shape ``(N, T_out, 50)``, or ``None``.
    times_ps
        Recorded sample times in picoseconds with shape ``(T_out,)``.
    """

    psi: np.ndarray
    n_active: np.ndarray | None
    n_inactive: np.ndarray | None
    times_ps: np.ndarray


def simulate_batch(
    resources: SharedSimResources,
    powers_batch: np.ndarray,
    warmup_ps: float,
    stride_steps: int,
    record_reservoir: bool,
    batch_size: int,
) -> BatchedTraces:
    """Simulate encoded images in GPU batches and return dynamic readout traces."""
    cfg = resources.cfg
    xp = compute_engine.xp
    p = np.asarray(powers_batch, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 49:
        raise ValueError(f"powers_batch shape must be (N, 49), got {p.shape}")
    if p.shape[0] == 0:
        raise ValueError("powers_batch must contain at least one sample")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not np.all(np.isfinite(p)):
        raise ValueError("powers_batch must be finite")
    if getattr(cfg.physics, "init_seed", None) is None:
        raise ValueError("simulate_batch requires physics.init_seed for deterministic init")
    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    if stride_steps <= 0:
        raise ValueError(f"stride_steps must be positive, got {stride_steps}")
    warmup_step = int(np.ceil(warmup_ps / cfg.solver.dt))
    output_steps = _output_steps(n_steps, warmup_step, int(stride_steps))
    if output_steps.size == 0:
        raise ValueError("warmup_ps and stride_steps leave no readout frames")
    n_samples = int(p.shape[0])
    n_out = int(output_steps.size)
    n_channels = int(resources.readout_masks.shape[0])
    factors_host = _pulse_time_factors(n_steps, float(cfg.solver.dt), resources.pulse_scalars)
    factors_device = xp.asarray(factors_host, dtype=xp.float64)
    nonzero_factors = factors_host != 0.0
    use_zero_branch = bool(np.any(~nonzero_factors))
    traces_psi = np.empty((n_samples, n_out, n_channels), dtype=np.float64)
    traces_n_active = (
        np.empty((n_samples, n_out, n_channels), dtype=np.float64)
        if record_reservoir
        else None
    )
    traces_n_inactive = (
        np.empty((n_samples, n_out, n_channels), dtype=np.float64)
        if record_reservoir
        else None
    )
    times_ps = (output_steps.astype(np.float64) + 1.0) * float(cfg.solver.dt)
    _log_memory_info(xp, batch_size)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        _simulate_chunk(
            resources,
            p[start:stop],
            factors_device,
            nonzero_factors,
            use_zero_branch,
            warmup_step,
            int(stride_steps),
            record_reservoir,
            traces_psi[start:stop],
            traces_n_active[start:stop] if traces_n_active is not None else None,
            traces_n_inactive[start:stop] if traces_n_inactive is not None else None,
        )
    return BatchedTraces(
        psi=traces_psi,
        n_active=traces_n_active,
        n_inactive=traces_n_inactive,
        times_ps=times_ps,
    )


def _simulate_chunk(
    resources: SharedSimResources,
    powers: np.ndarray,
    factors_device,
    nonzero_factors: np.ndarray,
    use_zero_branch: bool,
    warmup_step: int,
    stride_steps: int,
    record_reservoir: bool,
    traces_psi: np.ndarray,
    traces_n_active: np.ndarray | None,
    traces_n_inactive: np.ndarray | None,
) -> None:
    cfg = resources.cfg
    xp = compute_engine.xp
    batch_count = int(powers.shape[0])
    powers_device = xp.asarray(powers, dtype=xp.float64)
    spatial_pump = xp.tensordot(powers_device, resources.envelopes_stack, axes=1)
    pump_zero = xp.zeros_like(spatial_pump)
    reservoir = create_reservoir(
        cfg.reservoir,
        cfg.physics,
        resources.grid,
        precision=cfg.solver.precision,
    )
    reservoir_state = reservoir.get_state()
    reservoir.set_state(
        tuple(
            xp.zeros((batch_count, resources.grid.ny, resources.grid.nx), dtype=field.dtype)
            for field in reservoir_state
        )
    )
    psi0 = make_initial_psi(
        xp,
        resources.grid.ny,
        resources.grid.nx,
        eps=cfg.physics.init_eps,
        mode=getattr(cfg.physics, "init_mode", "filtered_complex_gaussian"),
        dx=resources.grid.dx,
        dy=resources.grid.dy,
        k_cutoff_um=getattr(cfg.physics, "init_k_cutoff_um", None),
        seed=int(cfg.physics.init_seed),
        cdtype=xp.complex128,
        rdtype=xp.float64,
    )
    state = SimulationState.__new__(SimulationState)
    state.psi = xp.array(xp.broadcast_to(psi0[None, :, :], spatial_pump.shape), copy=True)
    state.psi_k = None
    state.t = 0.0
    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    out_idx = 0
    for step in range(n_steps):
        pump_t = (
            factors_device[step] * spatial_pump
            if not use_zero_branch or nonzero_factors[step]
            else pump_zero
        )
        resources.solver.step(resources.potential, pump_t, reservoir, resources.bc, state)
        if step >= warmup_step and (step - warmup_step) % stride_steps == 0:
            state.materialize_psi(xp)
            psi = state.psi
            rho = psi.real * psi.real + psi.imag * psi.imag
            traces_psi[:, out_idx, :] = compute_engine.to_cpu(
                xp.einsum("kyx,byx->bk", resources.readout_masks, rho)
            )
            if record_reservoir:
                n_active, n_inactive = reservoir.get_reservoir_densities()
                if traces_n_active is not None:
                    traces_n_active[:, out_idx, :] = compute_engine.to_cpu(
                        xp.einsum("kyx,byx->bk", resources.readout_masks, n_active)
                    )
                if traces_n_inactive is not None:
                    traces_n_inactive[:, out_idx, :] = compute_engine.to_cpu(
                        xp.einsum("kyx,byx->bk", resources.readout_masks, n_inactive)
                    )
            out_idx += 1
    if out_idx != traces_psi.shape[1]:
        raise RuntimeError(f"Expected {traces_psi.shape[1]} trace frames, recorded {out_idx}")


def _log_memory_info(xp, batch_size: int) -> None:
    device = getattr(getattr(xp, "cuda", None), "Device", None)
    if device is None:
        return
    free_bytes, total_bytes = device().mem_info
    LOGGER.info(
        "GPU memory before batched simulation: free=%d total=%d batch_size=%d",
        int(free_bytes),
        int(total_bytes),
        int(batch_size),
    )
