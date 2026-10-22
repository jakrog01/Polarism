"""Absorption registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.boundary_conditions.absorption.absorption_strategy import AbsorptionStrategy

available_absorptions: Registry[type[AbsorptionStrategy]] = Registry("absorption")


def register_absorption(name: str) -> Callable[[type[AbsorptionStrategy]], type[AbsorptionStrategy]]:
    """Return a decorator that registers an absorption strategy under ``name``."""
    return available_absorptions.register(name)
