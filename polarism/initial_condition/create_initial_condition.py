"""Initial-condition factory helpers."""
from __future__ import annotations

from polarism.config.simulation_parameters import InitialConditionParameters
from polarism.initial_condition.initial_condition_registry import available_initial_conditions


def create_initial_condition(grid, cfg: InitialConditionParameters, xp, cdtype):
    """Return the deterministic start field, or ``None`` for the ``zero`` condition.

    Returning ``None`` lets the caller keep the noise-only field untouched.
    """
    factory = available_initial_conditions.lookup(cfg.type)
    if cfg.type == "zero":
        return None
    psi = factory(grid, cfg, xp, cdtype)
    if tuple(psi.shape) != tuple(grid.shape):
        raise ValueError(
            f"Initial condition '{cfg.type}' returned shape {tuple(psi.shape)}, "
            f"expected the grid shape {tuple(grid.shape)}"
        )
    return xp.asarray(psi, dtype=cdtype)
