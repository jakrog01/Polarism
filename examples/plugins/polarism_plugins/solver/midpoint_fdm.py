"""A small periodic finite-difference midpoint solver for plugin authors."""

from polarism.solver import AbstractSolver, register_solver


@register_solver("example-midpoint-fdm")
class MidpointFDMSolver(AbstractSolver):
    def __init__(self, config, grid):
        super().__init__(config)
        if grid.boundary != "periodic":
            raise ValueError("example-midpoint-fdm requires a periodic grid")
        if config.physics.kinetic_relaxation_eta != 0:
            raise ValueError("example-midpoint-fdm does not implement kinetic relaxation")
        self.dx2 = grid.dx**2
        self.dy2 = grid.dy**2

    def _laplacian(self, psi):
        xp = self.xp
        return (
            (xp.roll(psi, 1, -2) - 2 * psi + xp.roll(psi, -1, -2)) / self.dy2
            + (xp.roll(psi, 1, -1) - 2 * psi + xp.roll(psi, -1, -1)) / self.dx2
        )

    def _rhs(self, psi, fields, potential, pump_field, reservoir):
        p = self.physics
        xp = self.xp
        rho = xp.abs(psi) ** 2
        active = reservoir.get_active_density(fields)
        inactive = reservoir.get_inactive_density(fields)
        dfields = reservoir.get_derivatives(psi, pump_field, fields, rho)
        kinetic = -(p.hbar**2 / (2 * p.m_eff)) * self._laplacian(psi)
        energy = potential + p.g_C * rho + p.g_R * active + p.g_I * inactive
        gain = (p.R * active - p.gamma_C) / 2
        dpsi = (-1j / p.hbar) * (kinetic + energy * psi) + gain * psi
        return dpsi, dfields

    def step(self, potential, pump, reservoir, boundary_condition, state):
        dt = self.config.solver.dt
        psi0 = state.field
        fields0 = reservoir.get_state()

        k1_psi, k1_fields = self._rhs(
            psi0, fields0, potential, pump(state.t), reservoir
        )
        psi_mid = psi0 + 0.5 * dt * k1_psi
        fields_mid = tuple(
            field + 0.5 * dt * rate
            for field, rate in zip(fields0, k1_fields)
        )
        k2_psi, k2_fields = self._rhs(
            psi_mid, fields_mid, potential, pump(state.t + 0.5 * dt), reservoir
        )

        psi_new = boundary_condition.after_step_action(psi0 + dt * k2_psi)
        state.field = psi_new.astype(psi0.dtype, copy=False)
        reservoir.set_state(tuple(
            field + dt * rate for field, rate in zip(fields0, k2_fields)
        ))
