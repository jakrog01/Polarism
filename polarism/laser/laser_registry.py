"""Laser registry helpers."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.laser.abstract_laser import AbstractLaser

available_lasers: Registry[type[AbstractLaser]] = Registry("laser")


def register_laser(name: str) -> Callable[[type[AbstractLaser]], type[AbstractLaser]]:
    """Return a decorator that registers a laser under ``name``."""
    return available_lasers.register(name)
