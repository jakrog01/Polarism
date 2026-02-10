from __future__ import annotations
from types import FunctionType

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

    def compute(self, **context) -> tuple[np.ndarray, np.ndarray]:
        field = self.compute_fn(**context)
        reduced = self.reduce_dim_fn(field)
        return field, reduced
