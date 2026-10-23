"""Callable total pump field with optional separable laser terms."""
from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.laser.abstract_laser import AbstractLaser


def _scalar_settings(laser):
    return {key: value for key, value in vars(laser).items()
            if isinstance(value, (str, int, float, bool, type(None)))}


def evaluate_pump(pump, t):
    """Evaluate a vector coefficient with its member axis when present."""
    terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
    if terms is None and hasattr(pump, "separable_term"):
        term = pump.separable_term()
        terms = (term,) if term is not None else None
    if terms:
        coefficients = [coefficient(t) for _, coefficient in terms]
        if any(np.ndim(value) != 0 for value in coefficients):
            result = 0
            for (envelope, _), value in zip(terms, coefficients):
                xp = compute_engine.xp
                vector = xp.asarray(value)
                if vector.ndim:
                    vector = vector[:, None, None, None]
                result = result + envelope * vector
            return result
    return pump(t)


class PumpField:
    """Evaluate laser fields while exposing a single cached spatial envelope."""

    def __init__(self, lasers, grid, precision: str):
        self.lasers = lasers
        self.grid = grid
        self.precision = precision
        self.xp = compute_engine.xp

    def __call__(self, t: float):
        lasers = self.lasers
        X, Y = self.grid.X, self.grid.Y
        terms = self.separable_terms()
        if terms:
            coefficients = [coefficient(t) for _, coefficient in terms]
            if any(np.ndim(value) != 0 for value in coefficients):
                total = 0
                for (envelope, _), value in zip(terms, coefficients):
                    vector = self.xp.asarray(value)
                    if vector.ndim:
                        vector = vector[:, None, None, None]
                    total = total + envelope * vector
                return total
        if not lasers:
            return self.xp.zeros(X.shape, dtype=real_dtype(self.xp, self.precision))
        first = lasers[0].get_power(X, Y, t)
        if len(lasers) == 1:
            return first
        total = first + lasers[1].get_power(X, Y, t)
        for laser in lasers[2:]:
            total += laser.get_power(X, Y, t)
        return total

    def separable_term(self):
        """Return (envelope, coefficient function) for one standard laser."""
        if len(self.lasers) != 1:
            return None
        laser = self.lasers[0]
        if (type(laser).get_power is AbstractLaser.get_power
                and hasattr(laser, "_spatial_envelope")):
            return laser._spatial_envelope, laser.power_coefficient
        return None

    def separable_terms(self):
        """Return every standard laser's (envelope, coefficient), or None if any differs."""
        terms = []
        for laser in self.lasers:
            if (type(laser).get_power is not AbstractLaser.get_power
                    or not hasattr(laser, "_spatial_envelope")):
                return None
            terms.append((laser._spatial_envelope, laser.power_coefficient))
        return terms


class BatchedPumpField:
    """Combine matching laser slots across members into batched pump terms.

    A plugin overriding ``get_power`` uses a slower per-member call fallback.
    """

    def __init__(self, member_lasers, grid, precision):
        self.member_lasers = member_lasers
        self.lasers = member_lasers[0]
        self.grid = grid
        self.precision = precision
        self.xp = compute_engine.xp
        self._terms = self._build_terms()
        self.last_components = []
        self.capture_components = False

    def _build_terms(self):
        if len({len(lasers) for lasers in self.member_lasers}) != 1:
            return None
        terms = []
        for slot in zip(*self.member_lasers):
            if any(type(laser).get_power is not AbstractLaser.get_power
                   or not hasattr(laser, "_spatial_envelope") for laser in slot):
                return None
            envelopes = [laser._spatial_envelope for laser in slot]
            shared = all(bool(self.xp.array_equal(envelopes[0], item)) for item in envelopes[1:])
            envelope = envelopes[0] if shared else self.xp.stack(envelopes)[:, None]
            if all(type(laser) is type(slot[0]) and
                   _scalar_settings(laser) == _scalar_settings(slot[0])
                   for laser in slot[1:]):
                coefficient = slot[0].power_coefficient
            else:
                def coefficient(t, slot=slot, dtype=envelope.dtype):
                    if self.xp is np or len({type(laser) for laser in slot}) != 1:
                        return np.asarray([laser.power_coefficient(t) for laser in slot], dtype=dtype)
                    return type(slot[0]).batch_power_coefficient(slot, [t])[0].astype(dtype)
            terms.append((envelope, coefficient))
        return terms

    def separable_terms(self):
        """Return stacked envelopes and scalar or member-vector coefficients."""
        return self._terms

    def __call__(self, t):
        """Evaluate all members, including non-separable plugin lasers."""
        if self._terms is not None:
            total = 0
            components = [] if self.capture_components else None
            for envelope, coefficient in self._terms:
                value = self.xp.asarray(coefficient(t))
                if value.ndim:
                    value = value[:, None, None, None]
                component = envelope * value
                if components is not None:
                    components.append(component)
                total = total + component
            if components is not None:
                self.last_components = components
            return total
        return self.xp.stack([PumpField(lasers, self.grid, self.precision)(t)
                              for lasers in self.member_lasers])[:, None]
