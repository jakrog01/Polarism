"""Storage backend registry."""
from __future__ import annotations

import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.results.storage.base_storage import BaseStorage

available_storages: dict[str, type[BaseStorage]] = {}


def register_storage(name: str):
    """Register a result storage backend."""
    def decorator(cls: type[BaseStorage]) -> type[BaseStorage]:
        """Register the decorated class and return it."""
        existing = available_storages.get(name)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"Storage '{name}' is already registered; overriding {existing!r}.",
                stacklevel=2,
            )
        available_storages[name] = cls
        return cls

    return decorator
