"""Solver registry helpers."""
from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.solver.abstract_solver import AbstractSolver

available_solvers = {}


def register_solver(name: str):
    """Register solver."""
    def decorator(cls: type[AbstractSolver]) -> type[AbstractSolver]:
        """Register the decorated object and return it."""
        existing = available_solvers.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Solver '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_solvers[name] = cls
        return cls

    return decorator
