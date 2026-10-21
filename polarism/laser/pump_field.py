"""Callable total pump field with optional separable laser terms."""
from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.laser.abstract_laser import AbstractLaser


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
                    vector = vector[:, None, None]
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
                        vector = vector[:, None, None]
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
