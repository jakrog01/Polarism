"""Callable total pump field with an optional separable single-laser term."""
from __future__ import annotations

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.laser.abstract_laser import AbstractLaser


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
