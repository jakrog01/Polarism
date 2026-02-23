from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.reservoir.abstract_reservoir import AbstractReservoir
    from polarism.simulation_grid_2D import SimulationGrid2D


@register_solver("split-step-fft")
class SplitStepFFTSolver(AbstractSolver):
    _kinetic_propagator: Union[np.ndarray, cp.ndarray]

    def __init__(self, config: Config, grid: SimulationGrid2D):
        super().__init__(config)
        self._kinetic_propagator = self.xp.exp(
            -1j
            * self.config.physics.hbar
            * grid.k_squared
            * (self.config.solver.dt / 2)
            / (2 * self.config.physics.m_eff)
        )

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        self._half_step_kinetic(state)
        self._full_step_potential(pump, reservoir, potential, state)
        self._half_step_kinetic(state)
        state.psi = boundary_condition.after_step_action(state.psi)

    def _half_step_kinetic(self, state: SimulationState) -> None:
        psi_k = self.xp.fft.fft2(state.psi)
        psi_k *= self._kinetic_propagator
        state.psi = self.xp.fft.ifft2(psi_k)

    def _full_step_potential(
        self,
        P: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        potential: Union[np.ndarray, cp.ndarray],
        state: SimulationState,
    ) -> None:
        reservoir.step(self.config.solver.dt, state.psi, P)
        nR = reservoir.get_reservoir_density()
        eff_energy = (
            potential
            + self.config.physics.g_C * self.xp.abs(state.psi) ** 2
            + self.config.physics.g_R * nR
        )
        gain_loss = (self.config.physics.R * nR - self.config.physics.gamma_C) / 2.0
        state.psi = (
            state.psi
            * self.xp.exp(
                -1j * eff_energy * self.config.solver.dt / self.config.physics.hbar
            )
            * self.xp.exp(gain_loss * self.config.solver.dt)
        )
