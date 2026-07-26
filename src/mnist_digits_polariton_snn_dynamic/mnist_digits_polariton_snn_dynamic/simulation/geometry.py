"""Pump geometry providers for dynamic SNN simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class GeometryProvider(Protocol):
    """Pump-position provider consumed by shared simulation resources."""

    @property
    def n_spots(self) -> int:
        """Number of pump spots."""

    @property
    def positions_um(self) -> np.ndarray:
        """Pump-center positions with shape ``(n_spots, 2)``."""

    @property
    def sigma_space_um(self) -> float:
        """Gaussian spatial width in micrometers."""

    def required_half_span_um(self) -> float:
        """Return the half span needed inside the CAP-clear window."""

    def describe(self) -> dict[str, int | float | str]:
        """Return JSON-serializable geometry metadata."""


@dataclass(frozen=True, slots=True)
class GridLatticeGeometry:
    """Square grid-lattice pump geometry."""

    n_side: int
    pitch_um: float
    sigma_space_um: float
    center_x_um: float
    center_y_um: float

    def __post_init__(self) -> None:
        n_side = int(self.n_side)
        pitch_um = float(self.pitch_um)
        sigma_space_um = float(self.sigma_space_um)
        if n_side <= 0:
            raise ValueError(f"n_side must be positive, got {self.n_side}")
        if not np.isfinite(pitch_um) or not np.isfinite(sigma_space_um):
            raise ValueError("pitch_um and sigma_space_um must be finite")
        if sigma_space_um <= 0.0:
            raise ValueError(f"sigma_space_um must be positive, got {sigma_space_um}")
        if pitch_um < 5.0 * sigma_space_um:
            raise ValueError(
                "grid_lattice pitch must be at least 5 sigma_space values, "
                f"got pitch_um={pitch_um} and sigma_space_um={sigma_space_um}"
            )

    @property
    def n_spots(self) -> int:
        """Number of lattice spots."""
        return int(self.n_side) * int(self.n_side)

    @property
    def positions_um(self) -> np.ndarray:
        """Lattice positions with shape ``(n_spots, 2)``."""
        n_side = int(self.n_side)
        offsets = (
            np.arange(n_side, dtype=np.float64) - np.float64(n_side - 1) / np.float64(2.0)
        ) * np.float64(self.pitch_um)
        xs, ys = np.meshgrid(
            offsets + np.float64(self.center_x_um),
            offsets + np.float64(self.center_y_um),
            indexing="xy",
        )
        return np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64, copy=False)

    def required_half_span_um(self) -> float:
        """Return lattice half span plus Gaussian guard."""
        return float(
            (np.float64(self.n_side - 1) / np.float64(2.0)) * np.float64(self.pitch_um)
            + np.float64(4.0) * np.float64(self.sigma_space_um)
        )

    def describe(self) -> dict[str, int | float | str]:
        """Return JSON-serializable geometry metadata."""
        return {
            "type": "grid_lattice",
            "n_side": int(self.n_side),
            "n_spots": int(self.n_spots),
            "pitch_um": float(self.pitch_um),
            "sigma_space_um": float(self.sigma_space_um),
            "center_x_um": float(self.center_x_um),
            "center_y_um": float(self.center_y_um),
            "required_half_span_um": float(self.required_half_span_um()),
        }
