"""Absorption registry helpers."""
from __future__ import annotations

import warnings

available_boundary_conditions: dict[str, type] = {}


def register_absorption(name: str):
    """Register absorption."""
    def decorator(cls: type) -> type:
        """Register the decorated object and return it."""
        existing = available_boundary_conditions.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Absorption '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_boundary_conditions[name] = cls
        return cls

    return decorator
