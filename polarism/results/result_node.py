"""Result node definitions."""
from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, Any

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import numpy as np


class ResultNode:
    """Describe one computed result."""
    name: str
    compute_fn: FunctionType
    reduce_dim_fn: FunctionType
    cmap: str | None
    scaling: str | None
    clim: tuple[float, float] | None
    expose: bool
    save: bool
    cut: bool | None
    is_field: bool

    def __init__(
        self,
        name: str,
        compute_fn: FunctionType,
        reduce_dim_fn: FunctionType,
        cmap: str | None,
        scaling: str | None,
        clim: tuple[float, float] | None,
        expose: bool,
        save: bool,
        cut: bool | None,
        is_field: bool | None = None,
    ):
        """Store the functions and metadata for one result."""
        self.name = name
        self.compute_fn = compute_fn
        self.reduce_dim_fn = reduce_dim_fn
        self.cmap = cmap
        self.scaling = scaling
        self.clim = clim
        self.expose = expose
        self.save = save
        self.cut = cut
        self.is_field = is_field if is_field is not None else (cmap is not None)

    def compute_cpu(self, **context) -> tuple[np.ndarray, Any]:
        """Compute the field and reduced value on the CPU."""
        field_raw = self.compute_fn(**context)
        reduced_raw = self.reduce_dim_fn(field_raw)
        return compute_engine.to_cpu(field_raw), compute_engine.to_cpu(reduced_raw)

    def compute_field_cpu(self, **context) -> np.ndarray:
        """Compute the field value on the CPU."""
        field_raw = self.compute_fn(**context)
        return compute_engine.to_cpu(field_raw)

    def compute_reduced_cpu(self, **context) -> Any:
        """Compute the reduced value on the CPU."""
        field_raw = self.compute_fn(**context)
        reduced_raw = self.reduce_dim_fn(field_raw)
        return compute_engine.to_cpu(reduced_raw)
