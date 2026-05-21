"""GPU-accelerated false-color frame rendering for online animation.

All functions accept either NumPy or CuPy arrays via the ``xp`` argument.
Transforms are applied in-place-style (returning new arrays) so the caller
retains the original data unchanged.
"""
from __future__ import annotations

from typing import Any

import numpy as np


_PUMP_NORM_GAMMA = 0.3


def build_lut(xp: Any, cmap_name: str) -> Any:
    """Build a 256×3 uint8 LUT on the active compute backend.

    Parameters
    ----------
    xp : module
        NumPy or CuPy.
    cmap_name : str
        Matplotlib colormap name.
    """
    try:
        import matplotlib.cm as _cm
    except ImportError:
        raise RuntimeError("matplotlib is required for colormap LUT generation")
    cmap = _cm.get_cmap(cmap_name)
    lut_cpu = np.uint8(cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255)
    return xp.asarray(lut_cpu)


def apply_transform(xp: Any, raw: Any, transform: str | None) -> Any:
    """Apply a named field transform.

    Parameters
    ----------
    xp : module
        NumPy or CuPy.
    raw : array
        Input field (may be complex).
    transform : str or None
        ``"abs2"`` → |raw|², ``"kspace_log"`` → log₁₀ normalised k-space
        power after fftshift, ``None`` → real part if complex else pass-through.
    """
    if transform == "abs2":
        return xp.abs(raw) ** 2
    if transform == "kspace_log":
        shifted = xp.fft.fftshift(xp.fft.fft2(raw))
        power = xp.abs(shifted) ** 2
        peak = float(power.max())
        if peak > 0.0:
            return xp.log10(power / peak + 1e-10)
        return xp.zeros(raw.shape, dtype=xp.float64)
    if xp.iscomplexobj(raw):
        return raw.real.astype(xp.float64)
    return raw


def normalize_to_uint8(
    xp: Any,
    arr: Any,
    vmin: float,
    vmax: float,
    norm: str | None = None,
    norm_gamma: float = _PUMP_NORM_GAMMA,
) -> Any:
    """Map float array to [0, 255] uint8 using linear or power normalization.

    Parameters
    ----------
    xp : module
        NumPy or CuPy.
    arr : array
        Real float array to normalize.
    vmin, vmax : float
        Clipping bounds for the color range.
    norm : str or None
        ``"power"`` applies gamma correction; ``None`` uses linear mapping.
    norm_gamma : float
        Exponent for power normalization.
    """
    span = max(vmax - vmin, 1e-30)
    arr_norm = (xp.clip(arr, vmin, vmax) - vmin) / span
    if norm == "power":
        return xp.clip(arr_norm ** norm_gamma * 255.0, 0.0, 255.0).astype(xp.uint8)
    return xp.clip(arr_norm * 255.0, 0.0, 255.0).astype(xp.uint8)


def pad_even(xp: Any, frame: Any) -> Any:
    """Pad frame height and width to even numbers (required by most encoders)."""
    h, w = frame.shape[:2]
    ph = h + (h % 2)
    pw = w + (w % 2)
    if ph == h and pw == w:
        return frame
    padded = xp.zeros((ph, pw, 3), dtype=xp.uint8)
    padded[:h, :w] = frame
    return padded


def render_panels(
    xp: Any,
    transformed: list[Any],
    norms: list[str | None],
    norm_gammas: list[float],
    luts: list[Any],
    vmins: list[float],
    vmaxs: list[float],
) -> Any:
    """Normalize, colormap, and tile panels into a single uint8 RGB frame.

    Parameters
    ----------
    xp : module
        NumPy or CuPy.
    transformed : list of array
        One pre-transformed float array per panel (in panel order).
    norms : list of str or None
        Per-panel normalization type (``"power"`` or ``None``).
    norm_gammas : list of float
        Per-panel gamma for power normalization.
    luts : list of array
        Per-panel 256×3 uint8 LUT on the active backend.
    vmins, vmaxs : list of float
        Per-panel color range bounds.
    """
    panels = []
    for i, arr in enumerate(transformed):
        indices = normalize_to_uint8(xp, arr, vmins[i], vmaxs[i], norms[i], norm_gammas[i])
        panels.append(luts[i][indices])
    return pad_even(xp, xp.concatenate(panels, axis=1))
