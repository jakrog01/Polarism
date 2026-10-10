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
) -> AbstractReservoir:
    """Create reservoir."""
    reservoir_type = reservoir_config.reservoir_type
    if reservoir_type not in available_reservoirs:
        raise ValueError(
            f"Unknown reservoir type: '{reservoir_type}'. "
            f"Available: {list(available_reservoirs.keys())}"
        )
    return available_reservoirs[reservoir_type](reservoir_config, physics, grid, precision)
