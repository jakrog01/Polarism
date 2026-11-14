"""Solver factory helpers."""
from __future__ import annotations

from polarism.solver.solver_registry import available_solvers


def create_solver(config, grid, model):
    """Create a solver from the registered implementation."""
    return available_solvers.lookup(config.solver.method)(config, grid, model)
