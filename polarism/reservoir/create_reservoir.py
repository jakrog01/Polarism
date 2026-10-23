"""Reservoir factory helpers."""
from __future__ import annotations

import inspect

from polarism.config.batch_physics import BatchPhysics
from polarism.compute_engine import compute_engine

from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from polarism.reservoir.reservoir_registry import available_reservoirs
from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.grid.simulation_grid_2d import SimulationGrid2D


def create_reservoir(
    reservoir_config: ReservoirParameters,
    physics: PhysicsConstants,
    grid: SimulationGrid2D,
    precision: str = "double",
    *,
    batch_size: int | None = None,
    batch_physics=None,
) -> AbstractReservoir:
    """Create a reservoir with canonical zero fields shaped ``(B, 1, ny, nx)``."""
    reservoir_type = reservoir_config.reservoir_type

    batched = BatchPhysics(physics, batch_physics, compute_engine.xp, precision)
    component = available_reservoirs.lookup(reservoir_type)
    if batched.size is None:
        reservoir = component(reservoir_config, physics, grid, precision)
    else:
        parameters = inspect.signature(component).parameters
        if "batch_physics" not in parameters and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            raise ValueError(f"Reservoir {reservoir_type!r} does not accept batch_physics")
        reservoir = component(reservoir_config, physics, grid, precision,
                              batch_physics=batch_physics)
        if batch_size is None:
            batch_size = batched.size
        elif batch_size != batched.size:
            raise ValueError(f"batch_physics length {batched.size} does not match batch_size {batch_size}")
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not hasattr(reservoir, "reset"):
            raise ValueError(f"Reservoir {reservoir_type!r} does not support batched reset")
        reservoir.reset(batch_size=batch_size)
    return reservoir
