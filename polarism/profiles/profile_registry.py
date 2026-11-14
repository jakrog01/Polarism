"""Registration and validation of spatial profile functions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polarism.registry import Registry

ProfileFactory = Callable[[object, dict[str, Any], object], object]
available_profiles: Registry[ProfileFactory] = Registry("profile")


def register_profile(name: str, *, params: tuple[str, ...] = ()):
    """Register a ``(grid, params, xp)`` function and its accepted spec keys."""
    def decorator(factory: ProfileFactory) -> ProfileFactory:
        factory.params = tuple(params)
        return available_profiles.register(name)(factory)
    return decorator
