from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)
from polarism.config.simulation_parameters import PotentialParameters
from polarism.potential.potential_registy import available_potentials
from polarism.simulation_grid_2D import SimulationGrid2D


def create_potential(
    potential_config: PotentialParameters, grid: SimulationGrid2D
) -> AbsorptionStrategy:
    if potential_config.potential_type not in available_potentials:
        raise ValueError(
            f"Unknown potential: '{potential_config.potential_type}'. "
            f"Available: {list(available_potentials.keys())}"
        )

    return available_potentials[potential_config.potential_type](grid.X)
