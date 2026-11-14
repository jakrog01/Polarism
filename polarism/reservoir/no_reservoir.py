"""Reservoir-free condensate model."""
from __future__ import annotations

from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.reservoir_registry import register_reservoir


@register_reservoir("none")
class NoReservoir(AbstractReservoir):
    """Expose the reservoir interface with no density fields."""

    state_names = ()

    def __init__(self, reservoir_config, physics, grid, precision="double", batch_physics=None):
        self.config = reservoir_config
        self._batched = False

    def reset(self, batch_size=None):
        """Record whether the condensate is batched."""
        self._batched = batch_size is not None

    def step(self, dt, psi_mid, pump):
        """Advance no reservoir fields."""

    def get_reservoir_density(self):
        """Return zero density."""
        return 0

    def make_result_nodes(self):
        """Expose no reservoir result nodes."""
        return []

    def get_state(self):
        """Return the empty reservoir state."""
        return ()

    def set_state(self, state):
        """Accept only the empty reservoir state."""
        if state != ():
            raise ValueError("NoReservoir expects an empty state")

    def get_derivatives(self, psi, pump, state, rho=None):
        """Return no reservoir derivatives."""
        return ()

    def get_active_density(self, state):
        """Return zero active density."""
        return 0

    def get_inactive_density(self, state):
        """Return zero inactive density."""
        return 0
