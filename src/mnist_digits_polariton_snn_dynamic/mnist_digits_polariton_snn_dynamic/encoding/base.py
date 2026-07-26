"""Input encoders for dynamic SNN simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Input-to-pump encoder consumed by the SNN pipeline."""

    @property
    def n_spots(self) -> int:
        """Number of encoded pump channels."""

    def input_shape(self) -> tuple[int, ...]:
        """Return the per-sample input shape."""

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode input samples into pump powers."""

    def describe(self) -> dict[str, int | float | str]:
        """Return JSON-serializable encoder metadata."""


@dataclass(frozen=True, slots=True)
class LinearPixelEncoder:
    """Linear pixel-to-power encoder."""

    n_side: int
    power_min: float
    power_max: float

    def __post_init__(self) -> None:
        n_side = int(self.n_side)
        power_min = float(self.power_min)
        power_max = float(self.power_max)
        if n_side <= 0:
            raise ValueError(f"n_side must be positive, got {self.n_side}")
        if not np.isfinite(power_min) or not np.isfinite(power_max):
            raise ValueError("Power bounds must be finite")
        if not power_min < power_max:
            raise ValueError("power_min must be smaller than power_max")

    @property
    def n_spots(self) -> int:
        """Number of encoded pump channels."""
        return int(self.n_side) * int(self.n_side)

    def input_shape(self) -> tuple[int, ...]:
        """Return the expected per-sample image shape."""
        side = int(self.n_side)
        return (side, side)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Map image intensities linearly to pump powers."""
        images = np.asarray(inputs, dtype=np.float64)
        expected = self.input_shape()
        if images.ndim != 3 or images.shape[1:] != expected:
            raise ValueError(
                "Expected images with shape "
                f"(N, {expected[0]}, {expected[1]}), got {images.shape}"
            )
        if np.min(images) < 0.0 or np.max(images) > 1.0:
            raise ValueError("Image values must be in [0, 1]")
        powers = float(self.power_min) + images.reshape(images.shape[0], self.n_spots) * (
            float(self.power_max) - float(self.power_min)
        )
        return np.ascontiguousarray(powers, dtype=np.float64)

    def describe(self) -> dict[str, int | float | str]:
        """Return JSON-serializable encoder metadata."""
        return {
            "type": "linear_pixel",
            "n_side": int(self.n_side),
            "n_spots": int(self.n_spots),
            "power_min": float(self.power_min),
            "power_max": float(self.power_max),
        }
