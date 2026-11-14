"""Potential made from a registered spatial profile."""
from __future__ import annotations

from polarism.potential.potential_registry import register_potential
from polarism.profiles import create_profile


@register_potential("profile")
def profile_potential(X, Y, cfg, grid=None):
    """Return a profile whose amplitude is measured in meV."""
    if grid is None:
        raise ValueError("Profile potential requires a simulation grid")
    return create_profile(cfg.profile, grid)
