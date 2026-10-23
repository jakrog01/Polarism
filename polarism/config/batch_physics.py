"""Validated per-member physics values for solver and reservoir backends."""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from polarism.config.simulation_parameters import PhysicsConstants


_FLOAT_FIELDS = frozenset(
    field.name for field in fields(PhysicsConstants)
    if field.type is float or field.type == "float"
)


class BatchPhysics:
    """Hold one-dimensional host values and broadcast backend values."""

    def __init__(self, physics, values, xp, precision="double"):
        if isinstance(values, BatchPhysics):
            self.host = values.host
            self.size = values.size
        else:
            self.host = {}
            self.size = None
            for name, value in (values or {}).items():
                if name not in _FLOAT_FIELDS:
                    raise ValueError(f"Unknown or non-float batch_physics field: {name!r}")
                array = np.asarray(value.get() if hasattr(value, "get") else value,
                                   dtype=np.float64)
                if array.ndim != 1 or array.size == 0:
                    raise ValueError(f"batch_physics.{name} must be a nonempty 1-D array")
                if self.size is not None and array.size != self.size:
                    raise ValueError(f"batch_physics.{name} has length {array.size}; expected {self.size}")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"batch_physics.{name} must contain finite values")
                self.size = array.size
                self.host[name] = array
        self.physics = physics
        self.xp = xp
        dtype = np.float32 if precision == "single" else np.float64
        self.arrays = {name: xp.asarray(value, dtype=dtype).reshape(-1, 1, 1, 1)
                       for name, value in self.host.items()}

    def get(self, name):
        """Return a broadcast array if the field varies, otherwise the scalar."""
        return self.arrays.get(name, getattr(self.physics, name, 0.0))

    def any_nonzero(self, name):
        """Check whether an optional term is active in any member."""
        return bool(np.any(self.host[name])) if name in self.host else getattr(self.physics, name, 0.0) != 0.0

    def check(self, field):
        """Require a canonical scalar field with the matching batch axis."""
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError(f"batch_physics requires canonical field shape; got {field.shape}")
        if self.size is None:
            return
        actual = field.shape[0]
        if actual != self.size:
            raise ValueError(f"batch_physics length {self.size} requires state batch size {self.size}; got {actual}")
