"""Create a dimensionless spatial array from a registered profile."""
from __future__ import annotations

import math

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.profiles.profile_registry import available_profiles


def validate_profile_spec(spec: dict, ndim: int) -> None:
    """Check common keys and delegate built-in parameter validation."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("profile.type is required and must be a string")
    factory = available_profiles.lookup(spec["type"])
    unknown = set(spec) - {"type", "amplitude", "x0", "y0"} - set(factory.params)
    if unknown:
        raise ValueError(f"Unknown profile key(s): {sorted(unknown)}")
    for key in ("amplitude", "x0", "y0"):
        value = spec.get(key, 1.0 if key == "amplitude" else 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"profile.{key} must be a finite number")
    if spec["type"] in {"ring", "lattice"} and ndim != 2:
        raise ValueError(f"profile.type={spec['type']!r} requires a 2D grid")
    validator = getattr(factory, "validate", None)
    if validator is not None:
        validator(spec, ndim)


def create_profile(spec: dict, grid):
    """Return a real array of ``grid.shape`` on the active backend."""
    validate_profile_spec(spec, grid.ndim)
    xp = compute_engine.xp
    params = {**spec, "x0": spec.get("x0", 0.0), "y0": spec.get("y0", 0.0)}
    result = xp.asarray(available_profiles.lookup(spec["type"])(grid, params, xp))
    if result.shape != grid.shape:
        raise ValueError(f"Profile returned shape {result.shape}, expected {grid.shape}")
    if result.dtype.kind != "f":
        if result.dtype.kind not in "iu":
            raise ValueError("Profile must return a real array")
        result = result.astype(xp.float64)
    if not bool(xp.all(xp.isfinite(result))):
        raise ValueError("Profile must contain only finite values")
    return result * spec.get("amplitude", 1.0)
