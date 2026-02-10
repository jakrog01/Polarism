from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.laser.abstract_laser import AbstractLaser

available_lasers: dict[str, type[AbstractLaser]] = {}


def register_laser(name: str):
    def decorator(cls: type[AbstractLaser]) -> type[AbstractLaser]:
        available_lasers[name] = cls
        return cls

    return decorator
