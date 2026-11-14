"""Simulation model and finite-difference reference right-hand side."""
from __future__ import annotations

from polarism.compute_engine import compute_engine
from polarism.dispersion import ParabolicDispersion


class Model:
    """Collect the physical components passed to a solver."""

    def __init__(self, grid, physics, precision, potential, pump, reservoir,
                 boundary, noise=None, terms=(), dispersion=None):
        self.grid = grid
        self.physics = physics
        self.precision = precision
        self.xp = compute_engine.xp
        self.potential = potential
        self.pump = pump
        self.reservoir = reservoir
        self.boundary = boundary
        self.noise = noise
        self.terms = tuple(terms)
        self.dispersion = ParabolicDispersion() if dispersion is None else dispersion
        self._laplacian_buf = None

    def extra_coefficients(self, field, reservoir_fields, t):
        """Sum the extra terms into ``(energy, rate)`` coefficients.

        Call only when ``terms`` is not empty.
        """
        energy = rate = 0
        for term in self.terms:
            value = term.coefficient(field, self.reservoir, reservoir_fields, t)
            if term.category == "energy":
                energy = energy + value
            else:
                rate = rate + value
        return energy, rate

    def extra_rhs(self, field, reservoir_fields, t):
        """Return the extra terms' contribution to ``d field / dt``."""
        energy, rate = self.extra_coefficients(field, reservoir_fields, t)
        return (-1j / self.physics.get("hbar")) * energy * field + rate * field

    def rhs(self, field, reservoir_fields, t, out=None):
        """Evaluate the finite-difference reference right-hand side."""
        from polarism.laser.pump_field import evaluate_pump

        pump = evaluate_pump(self.pump, t)
        reservoir = self.reservoir
        physics = self.physics
        xp = self.xp
        res_derivs = reservoir.get_derivatives(field, pump, reservoir_fields)
        n_active = reservoir.get_active_density(reservoir_fields)
        n_inactive = reservoir.get_inactive_density(reservoir_fields)
        lap = out
        if lap is None:
            if self._laplacian_buf is None or self._laplacian_buf.shape != field.shape:
                self._laplacian_buf = xp.zeros_like(field)
            lap = self._laplacian_buf
        self._laplacian(field, lap)
        kinetic = -(physics.get("hbar")**2) / (2 * physics.get("m_eff")) * lap
        eff_energy = (
            self.potential
            + physics.get("g_C") * xp.abs(field) ** 2
            + physics.get("g_R") * n_active
            + physics.get("g_I") * n_inactive
        )
        gain_loss = (
            physics.get("R") * n_active - physics.get("gamma_C")
        ) / 2.0
        eta = physics.get("kinetic_relaxation_eta")
        hbar_over_2m = physics.get("hbar") / (2.0 * physics.get("m_eff"))
        dfield = (-1j / physics.get("hbar")) * (
            kinetic + eff_energy * field
        ) + gain_loss * field + eta * n_active * hbar_over_2m * lap
        if self.terms:
            dfield = dfield + self.extra_rhs(field, reservoir_fields, t)
        return dfield, res_derivs

    def _laplacian(self, field, out):
        """Apply the grid's finite-difference Laplacian."""
        xp = self.xp
        dx = self.grid.dx
        if self.grid.boundary == "periodic":
            if self.grid.ndim == 1:
                out[:] = (xp.roll(field, -1, axis=-1) - 2 * field
                          + xp.roll(field, 1, axis=-1)) / (dx**2)
            else:
                dy = self.grid.dy
                out[:] = (
                    xp.roll(field, -1, axis=-2) - 2 * field + xp.roll(field, 1, axis=-2)
                ) / (dy**2) + (
                    xp.roll(field, -1, axis=-1) - 2 * field + xp.roll(field, 1, axis=-1)
                ) / (dx**2)
            return
        if self.grid.boundary != "closed-interval":
            raise ValueError(f"Unknown grid_type: {self.grid.boundary}")
        if self.grid.ndim == 1:
            out[..., 1:-1] = (field[..., 2:] - 2 * field[..., 1:-1]
                              + field[..., :-2]) / (dx**2)
            out[..., 0] = 2 * (field[..., 1] - field[..., 0]) / (dx**2)
            out[..., -1] = 2 * (field[..., -2] - field[..., -1]) / (dx**2)
            return
        dy = self.grid.dy
        nx, ny = self.grid.nx, self.grid.ny
        out.fill(0)
        if ny > 2 and nx > 2:
            out[..., 1:-1, 1:-1] = (field[..., 2:, 1:-1] - 2 * field[..., 1:-1, 1:-1] + field[..., :-2, 1:-1]) / (dy**2) + (field[..., 1:-1, 2:] - 2 * field[..., 1:-1, 1:-1] + field[..., 1:-1, :-2]) / (dx**2)
        if nx > 2:
            out[..., 0, 1:-1] = 2 * (field[..., 1, 1:-1] - field[..., 0, 1:-1]) / (dy**2) + (field[..., 0, 2:] - 2 * field[..., 0, 1:-1] + field[..., 0, :-2]) / (dx**2)
            out[..., -1, 1:-1] = 2 * (field[..., -2, 1:-1] - field[..., -1, 1:-1]) / (dy**2) + (field[..., -1, 2:] - 2 * field[..., -1, 1:-1] + field[..., -1, :-2]) / (dx**2)
        if ny > 2:
            out[..., 1:-1, 0] = (field[..., 2:, 0] - 2 * field[..., 1:-1, 0] + field[..., :-2, 0]) / (dy**2) + 2 * (field[..., 1:-1, 1] - field[..., 1:-1, 0]) / (dx**2)
            out[..., 1:-1, -1] = (field[..., 2:, -1] - 2 * field[..., 1:-1, -1] + field[..., :-2, -1]) / (dy**2) + 2 * (field[..., 1:-1, -2] - field[..., 1:-1, -1]) / (dx**2)
        out[..., 0, 0] = 2 * (field[..., 1, 0] - field[..., 0, 0]) / (dy**2) + 2 * (field[..., 0, 1] - field[..., 0, 0]) / (dx**2)
        out[..., 0, -1] = 2 * (field[..., 1, -1] - field[..., 0, -1]) / (dy**2) + 2 * (field[..., 0, -2] - field[..., 0, -1]) / (dx**2)
        out[..., -1, 0] = 2 * (field[..., -2, 0] - field[..., -1, 0]) / (dy**2) + 2 * (field[..., -1, 1] - field[..., -1, 0]) / (dx**2)
        out[..., -1, -1] = 2 * (field[..., -2, -1] - field[..., -1, -1]) / (dy**2) + 2 * (field[..., -1, -2] - field[..., -1, -1]) / (dx**2)
