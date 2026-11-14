"""Laser with a registered spatial profile and temporal envelope."""
from __future__ import annotations

from types import SimpleNamespace

from polarism.envelopes import create_envelope
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registry import register_laser
from polarism.profiles import create_profile


def _coordinate_grid(X, Y):
    """Infer a box from coordinate arrays for direct LaserFactory callers."""
    ndim = X.ndim
    nx = X.shape[-1]
    dx = float((X.reshape(-1)[1] - X.reshape(-1)[0]).item()) if nx > 1 else 0.0
    lx = float((X.max() - X.min()).item()) + dx
    if ndim == 1:
        return SimpleNamespace(X=X, shape=X.shape, ndim=1, lx=lx)
    ny = X.shape[0]
    dy = float((Y[1, 0] - Y[0, 0]).item()) if ny > 1 else 0.0
    ly = float((Y.max() - Y.min()).item()) + dy
    return SimpleNamespace(X=X, Y=Y, shape=X.shape, ndim=2, lx=lx, ly=ly)


@register_laser("profile")
class ProfileLaser(AbstractLaser):
    """Evaluate ``P0 * profile(r) * envelope(t - delay)``."""

    def __init__(self, laser_config, X, Y, precision="double", grid=None):
        super().__init__(laser_config, X, Y, precision)
        if self.power_definition != "peak_amplitude":
            raise ValueError("laser_type='profile' requires power_definition='peak_amplitude'")
        if not laser_config.profile:
            raise ValueError("laser.profile is required for laser_type='profile'")
        self.grid = grid if grid is not None else _coordinate_grid(X, Y)
        self.profile = laser_config.profile
        self.envelope = create_envelope(laser_config.envelope)
        self._envelope_signature = repr(laser_config.envelope)
        self.time_independent = self.envelope.time_independent
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        return create_profile(self.profile, self.grid)

    def _P_time(self, t):
        return self.envelope(t)
