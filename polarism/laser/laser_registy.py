"""Laser registry helpers."""
from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.laser.abstract_laser import AbstractLaser

available_lasers: dict[str, type[AbstractLaser]] = {}


def register_laser(name: str):
    """Register laser."""
    def decorator(cls: type[AbstractLaser]) -> type[AbstractLaser]:
        """Register the decorated object and return it."""
        existing = available_lasers.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Laser '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_lasers[name] = cls
        return cls

    return decorator
