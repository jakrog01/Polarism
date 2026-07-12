"""Shared GPU simulation resources reused across samples."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import Config
from polarism.grid.create_grid import create_grid
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.laser.pulse_gaussian import PulseGaussian
from polarism.potential.create_potential import create_potential
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.create_solver import create_solver

from snn_dynamic.config.loader import GeometryConfig, PulseConfig
from snn_dynamic.simulation.lasers_49 import build_49_lasers, build_49_positions


@dataclass(frozen=True, slots=True)
class PulseScalars:
    """Host pulse-Gaussian scalars shared by every lattice spot.

    Attributes
    ----------
    phase
        First pulse center offset in picoseconds.
    temporal_integral
        Cutoff Gaussian temporal integral.
    pulse_separation
        Center-to-center pulse separation in picoseconds.
    n_pulses
        Number of emitted pulses.
    sigma_time
        Gaussian temporal width in picoseconds.
    cutoff_sigma
        Temporal cutoff measured in Gaussian widths.
    """

    phase: float
    temporal_integral: float
    pulse_separation: float
    n_pulses: int
    sigma_time: float
    cutoff_sigma: float


@dataclass(frozen=True, slots=True, init=False)
class SharedSimResources:
    """Reusable GPU resources for pulsed dynamic SNN simulations.

    Attributes
    ----------
    cfg
        Polarism simulation configuration; ``cfg.solver.precision`` must be
        ``"double"``.
    grid
        Simulation grid with device coordinate arrays.
    bc
        Boundary condition, providing the CAP absorber.
    potential
        Static potential + CAP field on device, shape ``(Ny, Nx)``.
    solver
        Compiled solver instance.
    lasers
        Length-49 list of ``PulseGaussian`` instances whose normalized spatial
        envelopes are invariant across samples.
    positions
        Host array ``(49, 2)`` float64 of pump centers in micrometers.
    envelopes_stack
        Device array ``(49, Ny, Nx)`` float64 of the 49 spatial envelopes,
        precomputed once and contracted per sample against the encoded
        per-pulse energies to form the static spatial pump field.
    readout_masks
        Device array ``(50, Ny, Nx)`` float64 with 49 per-spot disk masks and
        one CAP-clear global mask, each normalized to discrete unit weight.
    pulse_scalars
        Host temporal-envelope scalars used to evaluate the pulsed pump factor
        inside the step loop.
    """

    cfg: Config
    grid: SimulationGrid2D
    bc: BoundaryCondition
    potential: Any
    solver: AbstractSolver
    lasers: list[PulseGaussian]
    positions: np.ndarray
    envelopes_stack: Any
    readout_masks: Any
    pulse_scalars: PulseScalars

    def __init__(
        self,
        cfg: Config,
        geometry: GeometryConfig,
        pulse: PulseConfig,
        mask_radius_um: float = 3.0,
    ) -> None:
        """Build grid, boundary, potential, solver, lasers, and envelopes."""
        compute_engine.configure(cfg.compute_engine)
        xp = compute_engine.xp
        grid = create_grid(cfg.grid)
        _validate_lattice_geometry(cfg, geometry)
        bc = BoundaryCondition(grid, cfg.boundary_condition, cfg.physics)
        potential = create_potential(cfg.potential, grid)
        cap = bc.before_step_action()
        if xp.iscomplexobj(cap) and not xp.iscomplexobj(potential):
            potential = potential.astype(xp.complex128)
            cap = cap.astype(xp.complex128)
        potential = potential + cap
        solver = create_solver(cfg, grid)
        positions = build_49_positions(
            geometry.center_x_um,
            geometry.center_y_um,
            geometry.pitch_um,
        )
        lasers = build_49_lasers(
            positions,
            np.zeros(49, dtype=np.float64),
            geometry.sigma_space_um,
            pulse,
            grid,
            precision=cfg.solver.precision,
        )
        envelopes_stack = xp.stack(
            [laser._spatial_envelope.astype(xp.float64, copy=False) for laser in lasers],
            axis=0,
        )
        readout_masks = _build_readout_masks(cfg, grid, positions, float(mask_radius_um))
        pulse_scalars = PulseScalars(
            phase=float(pulse.cutoff_sigma) * float(pulse.sigma_time),
            temporal_integral=math.sqrt(2.0 * math.pi)
            * float(pulse.sigma_time)
            * math.erf(float(pulse.cutoff_sigma) / math.sqrt(2.0)),
            pulse_separation=float(pulse.pulse_separation),
            n_pulses=int(pulse.n_pulses),
            sigma_time=float(pulse.sigma_time),
            cutoff_sigma=float(pulse.cutoff_sigma),
        )
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "bc", bc)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "lasers", lasers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "envelopes_stack", envelopes_stack)
        object.__setattr__(self, "readout_masks", readout_masks)
        object.__setattr__(self, "pulse_scalars", pulse_scalars)


def _validate_lattice_geometry(cfg: Config, geometry: GeometryConfig) -> None:
    pitch_um = float(geometry.pitch_um)
    sigma_space_um = float(geometry.sigma_space_um)
    if pitch_um < 5.0 * sigma_space_um:
        raise ValueError(
            "7x7 lattice pitch must be at least 5 sigma_space values, "
            f"got pitch_um={pitch_um} and sigma_space_um={sigma_space_um}"
        )
    required_half_span = 3.0 * pitch_um + 4.0 * sigma_space_um
    clear_half_x = 0.5 * float(cfg.grid.lx) * (
        1.0 - float(cfg.boundary_condition.mask_width_percent)
    )
    clear_half_y = 0.5 * float(cfg.grid.ly) * (
        1.0 - float(cfg.boundary_condition.mask_width_percent)
    )
    if required_half_span > clear_half_x or required_half_span > clear_half_y:
        raise ValueError(
            "7x7 lattice plus Gaussian guard must fit inside the CAP-clear window, "
            f"got required_half_span={required_half_span}, "
            f"clear_half_x={clear_half_x}, clear_half_y={clear_half_y}"
        )


def _build_readout_masks(
    cfg: Config,
    grid: SimulationGrid2D,
    positions: np.ndarray,
    mask_radius_um: float,
) -> Any:
    if mask_radius_um <= 0.0:
        raise ValueError(f"mask_radius_um must be positive, got {mask_radius_um}")
    xp = compute_engine.xp
    pos = np.asarray(positions, dtype=np.float64)
    x0 = xp.asarray(pos[:, 0], dtype=xp.float64)[:, None, None]
    y0 = xp.asarray(pos[:, 1], dtype=xp.float64)[:, None, None]
    spot_masks = (
        (grid.X[None, :, :] - x0) ** 2 + (grid.Y[None, :, :] - y0) ** 2
        <= mask_radius_um**2
    ).astype(xp.float64)
    spot_sums = xp.sum(spot_masks, axis=(1, 2), keepdims=True)
    if bool(compute_engine.to_cpu(xp.any(spot_sums <= 0.0))):
        raise ValueError("Every per-spot readout mask must contain at least one grid cell")
    spot_masks = spot_masks / spot_sums
    half_x = 0.5 * float(cfg.grid.lx) * (
        1.0 - float(cfg.boundary_condition.mask_width_percent)
    )
    half_y = 0.5 * float(cfg.grid.ly) * (
        1.0 - float(cfg.boundary_condition.mask_width_percent)
    )
    global_mask = (
        (xp.abs(grid.X) <= half_x) & (xp.abs(grid.Y) <= half_y)
    ).astype(xp.float64)
    global_sum = xp.sum(global_mask)
    if bool(compute_engine.to_cpu(global_sum <= 0.0)):
        raise ValueError("Global readout mask must contain at least one grid cell")
    global_mask = global_mask / global_sum
    return xp.concatenate((spot_masks, global_mask[None, :, :]), axis=0)
