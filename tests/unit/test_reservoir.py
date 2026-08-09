from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.reservoir.double_reservoir import DoubleReservoir
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from polarism.reservoir.single_reservoir import SingleReservoir
from tests.unit.conftest import grid

TOL_MACHINE_F64 = 1e-12  # Exact algebraic identity in float64.


@pytest.mark.parametrize("kind,arity", [(SingleReservoir, 1), (DoubleReservoir, 2), (QuadraticDoubleReservoir, 2)])
def test_reservoir_state_order_derivatives_and_decay(kind, arity: int) -> None:
    compute_engine.xp = np
    g, physics = grid(8, 8), PhysicsConstants()
    reservoir = kind(ReservoirParameters(expose_results=False), physics, g)
    state = tuple(np.full((8, 8), i + 1.0) for i in range(arity))
    reservoir.set_state(state)
    state = reservoir.get_state()
    assert len(state) == arity and reservoir.get_active_density(state) is state[0]
    psi = np.full((8, 8), 1 + 2j)
    pump = np.arange(64, dtype=float).reshape(8, 8) / 100
    got = reservoir.get_derivatives(psi, pump, state)
    rho = np.abs(psi) ** 2
    if kind is SingleReservoir:
        expected = (pump - (physics.gamma_R + physics.R * rho) * state[0],)
        rates = (physics.gamma_R,)
    elif kind is DoubleReservoir:
        expected = (physics.R_IA * state[1] - (physics.gamma_A + physics.R_AI + physics.R * rho) * state[0], pump - (physics.gamma_I + physics.R_IA) * state[1] + physics.R_AI * state[0])
        rates = (physics.gamma_A + physics.R_AI + physics.R_IA, physics.gamma_I + physics.R_IA - physics.R_AI)
    else:
        expected = (physics.kappa * state[1] ** 2 - physics.gamma_R * state[0] - physics.R * state[0] * rho, pump - physics.kappa * state[1] ** 2 - physics.gamma_I * state[1])
        rates = (physics.gamma_R, physics.gamma_I)
    assert all(np.max(np.abs(a - b)) < TOL_MACHINE_F64 for a, b in zip(got, expected))
    reservoir.set_state(tuple(np.full((8, 8), 2.0) for _ in range(arity)))
    old = reservoir.get_state()
    dt = 1e-7
    zero = np.zeros((8, 8), dtype=np.complex128)
    reservoir.step(dt, zero, np.zeros((8, 8)))
    new = reservoir.get_state()
    assert all(np.all(component >= 0) for component in new)
    assert all(np.max(np.abs(component - previous)) < 1e-4 for component, previous in zip(new, old))


@pytest.mark.parametrize("grid_type", ["periodic", "closed-interval"])
@pytest.mark.parametrize("kind", [SingleReservoir, DoubleReservoir, QuadraticDoubleReservoir])
def test_zero_diffusion_is_formula_only(kind, grid_type: str) -> None:
    compute_engine.xp = np
    g = grid(8, 8, grid_type)
    physics = PhysicsConstants(reservoir_diffusion_R=0.0, reservoir_diffusion_A=0.0, reservoir_diffusion_I=0.0)
    reservoir = kind(ReservoirParameters(expose_results=False), physics, g)
    state = tuple(np.ones((8, 8)) for _ in reservoir.get_state())
    result = reservoir.get_derivatives(np.ones((8, 8), complex), np.ones((8, 8)), state)
    assert all(np.isfinite(x).all() for x in result)


def test_quadratic_reservoir_rejects_mismatched_pump_shape() -> None:
    compute_engine.xp = np
    reservoir = QuadraticDoubleReservoir(ReservoirParameters(expose_results=False), PhysicsConstants(), grid(8, 8))
    with pytest.raises(ValueError, match="pump and psi"):
        reservoir.get_derivatives(np.ones((8, 8), complex), np.ones((7, 8)), reservoir.get_state())
