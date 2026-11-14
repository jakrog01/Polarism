"""Solver factory helpers."""
from __future__ import annotations

from typing import TYPE_CHECKING
import inspect

from polarism.config.batch_physics import BatchPhysics
from polarism.compute_engine import compute_engine

from polarism.solver.solver_registry import available_solvers

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.solver.abstract_solver import AbstractSolver


def create_solver(config: Config, grid: SimulationGrid, batch_physics=None,
                  noise_members=None) -> AbstractSolver:
    """Create solver."""
    physics = BatchPhysics(config.physics, batch_physics, compute_engine.xp, getattr(config.solver, "precision", "double"), ndim=grid.ndim)
    component = available_solvers.lookup(config.solver.method)
    if physics.size is None:
        solver = component(config, grid)
    else:
        parameters = inspect.signature(component).parameters
        if "batch_physics" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            raise ValueError(f"Solver {config.solver.method!r} does not accept batch_physics")
        solver = component(config, grid, batch_physics=batch_physics)
    if config.noise.enabled and config.noise.stochastic == "wigner" and any(
        member.noise.stochastic_strength > 0 for member in (noise_members or [config])
    ):
        from polarism.noise import NoiseApplier
        from polarism.noise.stochastic import draw_seed

        if config.noise.seed is None:
            config.noise.seed = draw_seed()
        for member in noise_members or []:
            if member.noise.seed is None:
                member.noise.seed = config.noise.seed
        solver.noise = NoiseApplier(config, grid, noise_members)
    return solver
