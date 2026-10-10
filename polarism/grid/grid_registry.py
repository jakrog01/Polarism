"""Grid registry helpers."""
from __future__ import annotations

import warnings

GRID_BOUNDARIES = {"periodic", "closed-interval"}

available_grids: dict[str, type] = {}


def register_grid(name: str):
    """Register grid."""
    def decorator(cls: type) -> type:
        """Register the decorated object and return it."""
        boundary = getattr(cls, "boundary", None)
        if boundary not in GRID_BOUNDARIES:
            raise ValueError(
                f"Grid '{name}' must define a class attribute 'boundary' set to one of "
                f"{sorted(GRID_BOUNDARIES)}, got {boundary!r}."
            )
        existing = available_grids.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Grid '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_grids[name] = cls
        return cls

    return decorator
