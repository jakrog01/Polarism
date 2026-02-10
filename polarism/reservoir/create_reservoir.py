from __future__ import annotations

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.simulation_grid_2D import SimulationGrid2D


def create_reservoir(
    reservoir_config: ReservoirParameters,
    physics: PhysicsConstants,
    grid: SimulationGrid2D,
) -> AbstractReservoir:
    if reservoir_config.reservoir_type == "single":
        return SingleReservoir(reservoir_config, physics, grid)
    elif reservoir_config.reservoir_type == "double":
        return DoubleReservoir(reservoir_config, physics, grid)
    else:
        raise ValueError(f"Unknown reservoir type: {reservoir_config.reservoir_type}")
