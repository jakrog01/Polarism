from __future__ import annotations

from typing import TYPE_CHECKING

from polarism.solver.solver_registry import available_solvers

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config
    from polarism.simulation_grid_2D import SimulationGrid2D
    from polarism.solver.abstract_solver import AbstractSolver


def create_solver(config: Config, grid: SimulationGrid2D) -> AbstractSolver:
    if config.solver.method not in available_solvers:
        raise ValueError(
            f"Unknown solver method: '{config.solver.method}'. "
            f"Available: {list(available_solvers.keys())}"
        )

    return available_solvers[config.solver.method](config, grid)
