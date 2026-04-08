"""Dtype helpers for precision-aware array allocation."""
from __future__ import annotations

from typing import Any


def real_dtype(xp: Any, precision: str) -> Any:
    """Return the real dtype for the given backend and precision setting."""
    return xp.float32 if precision == "single" else xp.float64


def complex_dtype(xp: Any, precision: str) -> Any:
    """Return the complex dtype for the given backend and precision setting."""
    return xp.complex64 if precision == "single" else xp.complex128
