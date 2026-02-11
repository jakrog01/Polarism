from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING, Any

from polarism.compute_engine import compute_engine

if TYPE_CHECKING:
    import numpy as np


class ResultNode:
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

    def compute(self, **context) -> tuple[np.ndarray, Any]:
        field_raw = self.compute_fn(**context)
        reduced_raw = self.reduce_dim_fn(field_raw)

        field_cpu = compute_engine.to_cpu(field_raw)
        reduced_cpu = compute_engine.to_cpu(reduced_raw)

        return field_cpu, reduced_cpu
