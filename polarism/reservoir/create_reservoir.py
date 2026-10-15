"""Reservoir factory helpers."""
from __future__ import annotations

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from polarism.reservoir.reservoir_registry import available_reservoirs
from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.grid.simulation_grid_2d import SimulationGrid2D


def create_reservoir(
    reservoir_config: ReservoirParameters,
    physics: PhysicsConstants,
    grid: SimulationGrid2D,
    precision: str = "double",
    *,
    batch_size: int | None = None,
) -> AbstractReservoir:
    """Create a reservoir with zero fields, optionally shaped ``(B, ny, nx)``."""
    reservoir_type = reservoir_config.reservoir_type
    if reservoir_type not in available_reservoirs:
        raise ValueError(
            f"Unknown reservoir type: '{reservoir_type}'. "
            f"Available: {list(available_reservoirs.keys())}"
        )
    reservoir = available_reservoirs[reservoir_type](reservoir_config, physics, grid, precision)
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not hasattr(reservoir, "reset"):
            raise ValueError(f"Reservoir {reservoir_type!r} does not support batched reset")
        reservoir.reset(batch_size=batch_size)
    return reservoir
