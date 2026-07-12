"""Per-sample GPU simulation runner for the dynamic SNN pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.init_condition import make_initial_psi
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.simulation_state import SimulationState

from snn_dynamic.simulation.resources import PulseScalars, SharedSimResources


@dataclass(frozen=True, slots=True)
class SimulationTraces:
    """Dynamic readout traces for one encoded image.

    Attributes
    ----------
    psi
        Condensate density traces with shape ``(T_out, 50)``.
    n_active
        Active reservoir traces with shape ``(T_out, 50)``, or ``None``.
    n_inactive
        Inactive reservoir traces with shape ``(T_out, 50)``, or ``None``.
    times_ps
        Recorded sample times in picoseconds with shape ``(T_out,)``.
    """

    psi: np.ndarray
    n_active: np.ndarray | None
    n_inactive: np.ndarray | None
    times_ps: np.ndarray


def simulate_one_image(
    resources: SharedSimResources,
    powers: np.ndarray,
    warmup_ps: float,
    stride_steps: int,
    record_reservoir: bool,
) -> SimulationTraces:
    """Simulate one encoded image and return dynamic readout traces.

    Parameters
    ----------
    resources
        Shared GPU resources; the pump field is formed as
        ``_pulse_time_factor(t) * powers @ resources.envelopes_stack``.
    powers
        Float64 vector with shape ``(49,)``.
    warmup_ps
        Time before trace recording starts.
    stride_steps
        Step interval between recorded trace frames.
    record_reservoir
        Whether reservoir density traces are recorded.

    Returns
    -------
    SimulationTraces
        Host trace arrays and their sample times.
    """
    cfg = resources.cfg
    xp = compute_engine.xp
    p = np.asarray(powers, dtype=np.float64)
    if p.shape != (49,):
        raise ValueError(f"powers shape must be (49,), got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("powers must be finite")
    reservoir = create_reservoir(
        cfg.reservoir,
        cfg.physics,
        resources.grid,
        precision=cfg.solver.precision,
    )
    state = SimulationState.__new__(SimulationState)
    state.psi = make_initial_psi(
        xp,
        resources.grid.ny,
        resources.grid.nx,
        eps=cfg.physics.init_eps,
        mode=getattr(cfg.physics, "init_mode", "filtered_complex_gaussian"),
        dx=resources.grid.dx,
        dy=resources.grid.dy,
        k_cutoff_um=getattr(cfg.physics, "init_k_cutoff_um", None),
        seed=int(getattr(cfg.physics, "init_seed", 42) or 42),
        cdtype=xp.complex128,
        rdtype=xp.float64,
    )
    state.t = 0.0
    n_steps = int(cfg.solver.total_time / cfg.solver.dt)
    if stride_steps <= 0:
        raise ValueError(f"stride_steps must be positive, got {stride_steps}")
    warmup_step = int(np.ceil(warmup_ps / cfg.solver.dt))
    output_steps = _output_steps(n_steps, warmup_step, int(stride_steps))
    if output_steps.size == 0:
        raise ValueError("warmup_ps and stride_steps leave no readout frames")
    n_out = int(output_steps.size)
    n_channels = int(resources.readout_masks.shape[0])
    p_device = xp.asarray(p, dtype=xp.float64)
    spatial_pump = xp.tensordot(p_device, resources.envelopes_stack, axes=1)
    traces_psi = xp.empty((n_out, n_channels), dtype=xp.float64)
    traces_n_active = (
        xp.empty((n_out, n_channels), dtype=xp.float64) if record_reservoir else None
    )
    traces_n_inactive = (
        xp.empty((n_out, n_channels), dtype=xp.float64) if record_reservoir else None
    )
    times_ps = np.empty(n_out, dtype=np.float64)
    out_idx = 0
    for step in range(n_steps):
        t = float(step) * float(cfg.solver.dt)
        pump_t = _pulse_time_factor(t, resources.pulse_scalars) * spatial_pump
        resources.solver.step(resources.potential, pump_t, reservoir, resources.bc, state)
        if step >= warmup_step and (step - warmup_step) % stride_steps == 0:
            psi = state.psi
            rho = psi.real * psi.real + psi.imag * psi.imag
            traces_psi[out_idx] = xp.tensordot(resources.readout_masks, rho, axes=2)
            if record_reservoir:
                n_active, n_inactive = reservoir.get_reservoir_densities()
                traces_n_active[out_idx] = xp.tensordot(
                    resources.readout_masks, n_active, axes=2
                )
                traces_n_inactive[out_idx] = xp.tensordot(
                    resources.readout_masks, n_inactive, axes=2
                )
            times_ps[out_idx] = (float(step) + 1.0) * float(cfg.solver.dt)
            out_idx += 1
    if out_idx != n_out:
        raise RuntimeError(f"Expected {n_out} trace frames, recorded {out_idx}")
    psi_host = np.asarray(compute_engine.to_cpu(traces_psi), dtype=np.float64)
    n_active_host = (
        np.asarray(compute_engine.to_cpu(traces_n_active), dtype=np.float64)
        if traces_n_active is not None
        else None
    )
    n_inactive_host = (
        np.asarray(compute_engine.to_cpu(traces_n_inactive), dtype=np.float64)
        if traces_n_inactive is not None
        else None
    )
    return SimulationTraces(
        psi=psi_host,
        n_active=n_active_host,
        n_inactive=n_inactive_host,
        times_ps=times_ps,
    )


def _pulse_time_factor(t: float, pulse_scalars: PulseScalars) -> float:
    n = round((t - pulse_scalars.phase) / pulse_scalars.pulse_separation)
    if pulse_scalars.n_pulses > 0 and n >= pulse_scalars.n_pulses:
        return 0.0
    dt_local = t - n * pulse_scalars.pulse_separation - pulse_scalars.phase
    if abs(dt_local) > pulse_scalars.cutoff_sigma * pulse_scalars.sigma_time:
        return 0.0
    return float(
        np.exp(-0.5 * (dt_local / pulse_scalars.sigma_time) ** 2)
        / pulse_scalars.temporal_integral
    )


def _output_steps(n_steps: int, warmup_step: int, stride_steps: int) -> np.ndarray:
    if warmup_step >= n_steps:
        return np.empty(0, dtype=np.int64)
    return np.arange(warmup_step, n_steps, stride_steps, dtype=np.int64)
