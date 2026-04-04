"""Absorption registry helpers."""
from __future__ import annotations

available_boundary_conditions: dict[str, type] = {}


def register_absorption(name: str):
    """Register absorption."""
    def decorator(cls: type) -> type:
        """Register the decorated object and return it."""
        available_boundary_conditions[name] = cls
        return cls

    return decorator
