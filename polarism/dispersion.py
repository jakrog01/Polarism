"""Open registry of single-component dispersion relations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from polarism.registry import Registry

available_dispersions: Registry[type["Dispersion"]] = Registry("dispersion")


def register_dispersion(name: str) -> Callable[[type["Dispersion"]], type["Dispersion"]]:
    """Return a decorator that registers a dispersion and sets its ``kind``."""
    def decorator(component):
        """Register ``component`` and record its name."""
        component.kind = name
        return available_dispersions.register(name)(component)

    return decorator


class Dispersion(ABC):
    """Provide the kinetic energy E(k) on the FFT grid."""

    kind: ClassVar[str]

    @abstractmethod
    def energy(self, grid, physics):
        """Return real E(k) in meV, broadcastable to the field shape."""
        raise NotImplementedError


@register_dispersion("parabolic")
class ParabolicDispersion(Dispersion):
    """Return ``hbar^2 k^2 / (2 m_eff)``."""

    def energy(self, grid, physics):
        """Return the parabolic kinetic energy."""
        return physics.get("hbar") ** 2 * grid.k_squared / (2 * physics.get("m_eff"))
