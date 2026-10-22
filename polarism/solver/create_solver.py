"""Solver factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING
import inspect

from polarism.config.batch_physics import BatchPhysics
from polarism.compute_engine import compute_engine

from polarism.solver.solver_registry import available_solvers

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid_2d import SimulationGrid2D
    from polarism.solver.abstract_solver import AbstractSolver


def create_solver(config: Config, grid: SimulationGrid2D, batch_physics=None) -> AbstractSolver:
    """Create solver."""
    physics = BatchPhysics(config.physics, batch_physics, compute_engine.xp, getattr(config.solver, "precision", "double"))
    component = available_solvers.lookup(config.solver.method)
    if physics.size is None:
        return component(config, grid)
    parameters = inspect.signature(component).parameters
    if "batch_physics" not in parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        raise ValueError(f"Solver {config.solver.method!r} does not accept batch_physics")
    return component(config, grid, batch_physics=batch_physics)
