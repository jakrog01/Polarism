"""Pulsed Gaussian laser model."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser

if TYPE_CHECKING:
    import numpy as np
    import cupy as cp

_VALID_POWER_DEFINITIONS = frozenset({"peak_amplitude", "pulse_energy"})


@register_laser("pulse-gaussian")
class PulseGaussian(AbstractLaser):
    """Represent a pulse gaussian."""
    sigma_space: float
    Pmax: float
    sigma_time: float
    P0: float
    pulse_separation: float
    cutoff_sigma: float
    n_pulses: int
    power_definition: str
    spatial_integral: float | None
    temporal_integral: float | None

    def __init__(self, laser_config: LaserParameters, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray], precision: str = "double"):
        """Set up the pulse gaussian."""
        super().__init__(laser_config, X, Y, precision)
        self.sigma_space = laser_config.sigma_space
        self.Pmax = laser_config.Pmax
        self.sigma_time = laser_config.sigma_time
        self.pulse_separation = laser_config.pulse_separation
        self.P0 = laser_config.P0
        self.cutoff_sigma = laser_config.cutoff_sigma
        self.n_pulses = laser_config.n_pulses
        self.phase = self.cutoff_sigma * self.sigma_time
        self.power_definition = getattr(laser_config, "power_definition", "peak_amplitude")
        if self.power_definition not in _VALID_POWER_DEFINITIONS:
            raise ValueError(
                f"power_definition={self.power_definition!r} is not valid; "
                f"must be one of {sorted(_VALID_POWER_DEFINITIONS)}"
            )

        if self.power_definition == "pulse_energy":
            dx = float(X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 1.0
            dy = float(Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 1.0
            raw_space = self._P_space(X, Y).astype(self._real_dtype, copy=False)
            self.spatial_integral = float(self.xp.sum(raw_space)) * dx * dy
            self.temporal_integral = (
                math.sqrt(2.0 * math.pi) * self.sigma_time
                * math.erf(self.cutoff_sigma / math.sqrt(2.0))
            )
            if not (math.isfinite(self.spatial_integral) and self.spatial_integral > 0):
                raise ValueError(
                    f"spatial_integral={self.spatial_integral} must be finite and positive; "
                    "check grid dimensions and sigma_space"
                )
            if not (math.isfinite(self.temporal_integral) and self.temporal_integral > 0):
                raise ValueError(
                    f"temporal_integral={self.temporal_integral} must be finite and positive; "
                    "check sigma_time and cutoff_sigma"
                )
            self._spatial_envelope = raw_space / self.spatial_integral
        else:
            self.spatial_integral = None
            self.temporal_integral = None
            self._finalize_spatial_envelope(X, Y)

    def _pulse_index(self, t: float) -> int:
        return max(0, round((t - self.phase) / self.pulse_separation))

    def _amplitude(self, t: float) -> float:
        """Return the base laser amplitude at time t."""
        n = self._pulse_index(t)
        if self.n_pulses > 0 and n >= self.n_pulses:
            return 0.0
        return min(self.P0 + n * (self.Pmax - self.P0), self.Pmax)

    def _P_space(self, X: Union[np.ndarray, cp.ndarray], Y: Union[np.ndarray, cp.ndarray]) -> Union[np.ndarray, cp.ndarray]:
        """Return the unnormalized spatial Gaussian profile."""
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return self.xp.exp(-0.5 * r2 / self.sigma_space**2)

    def _P_time(self, t: float) -> float:
        """Return the temporal pump envelope at time t.

        In peak_amplitude mode returns the raw Gaussian value in [0, 1].
        In pulse_energy mode returns the value divided by the temporal
        integral so that integrating over one pulse gives unity.
        """
        n = self._pulse_index(t)
        if self.n_pulses > 0 and n >= self.n_pulses:
            return 0.0
        dt = t - n * self.pulse_separation - self.phase
        if abs(dt) > self.cutoff_sigma * self.sigma_time:
            return 0.0
        raw = math.exp(-0.5 * (dt / self.sigma_time) ** 2)
        if self.power_definition == "pulse_energy":
            return raw / self.temporal_integral
        return raw

    @property
    def peak_amplitude_at_center(self) -> float:
        """Peak source density at the Gaussian center for the first pulse."""
        if self.power_definition == "pulse_energy":
            return self.P0 / (self.spatial_integral * self.temporal_integral)
        return self.P0
