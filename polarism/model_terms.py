"""Open registry of extra terms added to the condensate equation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from polarism.registry import Registry

available_terms: Registry[type["Term"]] = Registry("term")


def register_term(name: str) -> Callable[[type["Term"]], type["Term"]]:
    """Return a decorator that registers a term under ``name``."""
    return available_terms.register(name)


class Term(ABC):
    """Contribute a real coefficient to the condensate equation.

    An ``"energy"`` term adds ``(-i/hbar) * E * psi`` to ``d psi / dt`` and a
    ``"rate"`` term adds ``G * psi``. ``coefficient`` returns the real array
    ``E`` or ``G`` evaluated from the current stage values.
    """

    category: ClassVar[str]

    def __init__(self, params, grid, physics):
        """Store the term parameters, grid, and batched physics."""
        if self.category not in ("energy", "rate"):
            raise ValueError(f"Term category must be 'energy' or 'rate', got {self.category!r}")
        self.params = params
        self.grid = grid
        self.physics = physics

    @abstractmethod
    def coefficient(self, field, reservoir, reservoir_fields, t):
        """Return a real array broadcastable to ``field``."""
        raise NotImplementedError
