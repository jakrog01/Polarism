from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.solver.abstract_solver import AbstractSolver

available_solvers = {}


def register_solver(name: str):
    def decorator(cls: type[AbstractSolver]) -> type[AbstractSolver]:
        available_solvers[name] = cls
        return cls

    return decorator
