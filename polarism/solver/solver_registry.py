"""Solver registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.solver.abstract_solver import AbstractSolver

available_solvers: Registry[type[AbstractSolver]] = Registry("solver")


def register_solver(name: str) -> Callable[[type[AbstractSolver]], type[AbstractSolver]]:
    """Return a decorator that registers a solver under ``name``."""
    return available_solvers.register(name)
