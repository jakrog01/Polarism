"""Circular ROI scalar metrics for polariton HPC pipeline simulations."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class CircleROI:
    """Precomputed circular region of interest."""

    id: str
    x0: float
    y0: float
    radius: float
    mask: Any
    count: int
    cell_area: float

    @property
    def scalar_prefix(self) -> str:
        return f"roi_{_safe_id(self.id)}_circle"


def _safe_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    text = text.strip("_")
    return text or "roi"


def compile_circle_rois(roi_specs: list[dict[str, Any]], grid: Any, xp: Any) -> list[CircleROI]:
    """Precompute ROI masks on the active compute backend."""
    rois: list[CircleROI] = []
    for i, spec in enumerate(roi_specs):
        shape = str(spec.get("shape", "circle"))
        if shape != "circle":
            raise ValueError(f"Unsupported ROI shape {shape!r}; only 'circle' is supported")

        roi_id = _safe_id(str(spec.get("id", f"roi_{i}")))
        x0 = float(spec.get("x0", 0.0))
        y0 = float(spec.get("y0", 0.0))
        radius = float(spec["radius"])
        if radius <= 0.0:
            raise ValueError(f"ROI '{roi_id}' radius must be positive, got {radius}")

        mask = ((grid.X - x0) ** 2 + (grid.Y - y0) ** 2) <= radius**2
        count = int(float(xp.sum(mask)))
        if count < 1:
            raise ValueError(
                f"ROI '{roi_id}' contains no grid cells: center=({x0}, {y0}), "
                f"radius={radius}, dx={grid.dx}, dy={grid.dy}"
            )
        rois.append(
            CircleROI(
                id=roi_id,
                x0=x0,
                y0=y0,
                radius=radius,
                mask=mask.astype(xp.float64),
                count=count,
                cell_area=float(grid.dx * grid.dy),
            )
        )
    return rois


def scalar_names_for_rois(rois: list[CircleROI]) -> list[str]:
    """Return all scalar sidecar names emitted for the configured ROIs."""
    names: list[str] = []
    for roi in rois:
        prefix = roi.scalar_prefix
        names.extend(
            [
                f"{prefix}_mean_psi_sq",
                f"{prefix}_integral_psi_sq",
                f"{prefix}_mean_nR",
                f"{prefix}_mean_nI",
                f"{prefix}_integral_emission",
            ]
        )
    return names


def compute_roi_scalars(
    rois: list[CircleROI],
    psi_sq: Any,
    nR: Any,
    nI: Any,
    gamma_C: float,
    xp: Any,
) -> dict[str, float]:
    """Compute ROI scalar metrics.

    The active reservoir field is named ``nR`` in the emitted metrics even
    though the HDF5 field remains ``nA`` for backward compatibility.
    ``integral_emission`` is the radiative proxy ``gamma_C * integral |psi|^2``.
    """
    out: dict[str, float] = {}
    for roi in rois:
        prefix = roi.scalar_prefix
        psi_sum = float(xp.sum(psi_sq * roi.mask))
        nR_sum = float(xp.sum(nR * roi.mask))
        nI_sum = float(xp.sum(nI * roi.mask))
        psi_integral = psi_sum * roi.cell_area
        out[f"{prefix}_mean_psi_sq"] = psi_sum / roi.count
        out[f"{prefix}_integral_psi_sq"] = psi_integral
        out[f"{prefix}_mean_nR"] = nR_sum / roi.count
        out[f"{prefix}_mean_nI"] = nI_sum / roi.count
        out[f"{prefix}_integral_emission"] = float(gamma_C) * psi_integral
    return out
