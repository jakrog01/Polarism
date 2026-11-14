"""Registration of scalar temporal envelopes."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from polarism.registry import Registry

EnvelopeFactory = Callable[[dict[str, Any]], Callable[[float], float]]
available_envelopes: Registry[EnvelopeFactory] = Registry("envelope")


def register_envelope(name: str, *, params: tuple[str, ...] = ()):
    """Register a spec-to-callable factory and its accepted spec keys."""
    def decorator(factory: EnvelopeFactory) -> EnvelopeFactory:
        factory.params = tuple(params)
        return available_envelopes.register(name)(factory)
    return decorator
