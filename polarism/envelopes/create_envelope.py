"""Build a host-only scalar time function from an envelope spec."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from polarism.envelopes.envelope_registry import available_envelopes


@dataclass(frozen=True)
class Envelope:
    """Callable scalar envelope with a static-time marker for pump solvers."""
    function: Callable[[float], float]
    time_independent: bool = False

    def __call__(self, t: float) -> float:
        return float(self.function(t))


def create_envelope(spec: dict) -> Envelope:
    """Validate and construct a registered envelope."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("envelope.type is required and must be a string")
    factory = available_envelopes.lookup(spec["type"])
    unknown = set(spec) - {"type"} - set(factory.params)
    if unknown:
        raise ValueError(f"Unknown envelope key(s): {sorted(unknown)}")
    function = factory(spec)
    if not callable(function):
        raise ValueError("Envelope factory must return a callable")
    return Envelope(function, bool(getattr(function, "time_independent", False)))
