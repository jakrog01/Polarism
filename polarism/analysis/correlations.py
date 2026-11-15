"""Correlation functions derived from linear accumulated moments."""
from __future__ import annotations

import numpy as np


def _module(value):
    return __import__("cupy") if hasattr(value, "__cuda_array_interface__") else np


def g1(n_k, *, ndim=None):
    """Return the normalized, centered first-order spatial correlation."""
    xp = _module(n_k)
    n_k = xp.asarray(n_k)
    spatial_axes = tuple(range(-(ndim or min(n_k.ndim, 2)), 0))
    correlation = xp.fft.ifftn(xp.fft.ifftshift(n_k, axes=spatial_axes), axes=spatial_axes)
    origin = xp.asarray(correlation[(...,) + (0,) * len(spatial_axes)])
    denominator = xp.where(xp.abs(origin) > 1e-30, origin, 1)
    return xp.fft.fftshift(xp.where(xp.abs(origin)[(...,) + (None,) * len(spatial_axes)] > 1e-30,
                                     correlation / denominator[(...,) + (None,) * len(spatial_axes)], 0),
                           axes=spatial_axes)


def g2(mean, squared_mean):
    """Return the second-order correlation, zero at negligible density."""
    xp = _module(mean)
    mean = xp.asarray(mean)
    squared_mean = xp.asarray(squared_mean)
    return xp.where(mean > 1e-30, squared_mean / xp.maximum(mean * mean, 1e-60), 0)
