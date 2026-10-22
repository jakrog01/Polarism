"""Storage backend registry."""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from polarism.registry import Registry

if TYPE_CHECKING:
    from polarism.results.storage.base_storage import BaseStorage

available_storages: Registry[type[BaseStorage]] = Registry("storage")


def register_storage(name: str) -> Callable[[type[BaseStorage]], type[BaseStorage]]:
    """Return a decorator that registers a storage backend under ``name``."""
    return available_storages.register(name)
