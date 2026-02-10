from __future__ import annotations

from types import FunctionType

available_potentials = {}


def register_potential(name: str):
    def decorator(func: FunctionType) -> FunctionType:
        available_potentials[name] = func
        return func

    return decorator
