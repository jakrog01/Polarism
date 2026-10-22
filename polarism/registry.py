"""Shared component registration and lookup."""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry(dict[str, T]):
    """Map component names to implementations."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers an implementation."""
        def decorator(component: T) -> T:
            existing = self.get(name)
            if existing is not None and existing is not component:
                warnings.warn(
                    f"{self.kind.title()} '{name}' is already registered; overriding {existing!r}.",
                    stacklevel=2,
                )
            self[name] = component
            return component

        return decorator

    def lookup(self, name: str) -> T:
        """Return a registered implementation or raise a uniform error."""
        try:
            return self[name]
        except KeyError as exc:
            raise ValueError(f"Unknown {self.kind} '{name}'. Available: {sorted(self)}") from exc
