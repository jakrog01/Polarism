"""Simulation model and finite-difference reference right-hand side."""
from __future__ import annotations

from dataclasses import dataclass

from polarism.compute_engine import compute_engine
from polarism.dispersion import ParabolicDispersion


@dataclass(frozen=True)
class CoherentDrive:
    """Hold F(r, t) = spatial * envelope(t) and the frame energy E_p.

    The source in dψ/dt is -i F(r, t)/ħ, while the frame contributes
    +i E_p ψ/ħ to the same derivative.
    """
    spatial: object
    envelope: object
    energy: object


class Model:
    """Collect the physical components passed to a solver."""

    def __init__(self, grid, physics, precision, potential, pump, reservoir,
                 boundary, noise=None, terms=(), dispersion=None, drive=None,
                 potential_static=None, cap=None, potential_envelope=None,
                 pump_weights=None, reservoir_components=1):
        self.grid = grid
        self.physics = physics
        self.precision = precision
        self.xp = compute_engine.xp
        self.potential = potential
        self.potential_static = potential_static
        self.cap = cap
        self.potential_envelope = potential_envelope
        self.time_dependent_potential = potential_envelope is not None
        self.drive = drive
        self._has_drive = drive is not None
        self.pump = pump
        self.reservoir = reservoir
        self.boundary = boundary
        self.noise = noise
        self.terms = tuple(terms)
        self.dispersion = ParabolicDispersion() if dispersion is None else dispersion
        self._laplacian_buf = None
        self.components = 2 if getattr(physics.physics, "spin", "scalar") == "spinor" else 1
        self.pump_weights = pump_weights
        self.reservoir_components = reservoir_components

    def potential_at(self, t):
        """Return V(r) f(t) + CAP, keeping the absorber time independent."""
        if not self.time_dependent_potential:
            return self.potential
        return self.potential_static * self.potential_envelope(t) + self.cap

    def drive_rhs(self, t):
        """Return -i F(r, t)/ħ for the enabled coherent drive."""
        return (-1j / self.physics.get("hbar")) * self.drive.spatial * self.drive.envelope(t)

    def drive_contribution(self, field, t):
        """Return (+i E_p/ħ)ψ - i F(r, t)/ħ.

        For V = g = k = 0 without reservoir, dψ/dt =
        (i E_p/ħ - γ_C/2)ψ - i F/ħ, so
        ψ_ss = (-i F/ħ)/(γ_C/2 - i E_p/ħ).
        """
        return (1j / self.physics.get("hbar")) * self.drive.energy * field + self.drive_rhs(t)

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
        if self.components == 2:
            return self._spinor_rhs(field, reservoir_fields, t, out)
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
            (self.potential_at(t) if self.time_dependent_potential else self.potential)
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
        if self._has_drive:
            dfield = dfield + self.drive_contribution(field, t)
        return dfield, res_derivs

    def _spinor_rhs(self, field, reservoir_fields, t, out=None):
        """Evaluate circular-basis spinor dynamics.

        The TE–TM contribution is ``-β(∂x ∓ i∂y)² ψ∓`` in energy,
        giving ``+β(kx ∓ i ky)²`` off diagonal in Fourier space.
        """
        from polarism.laser.pump_field import evaluate_pump

        xp = self.xp
        physics = self.physics
        pump = evaluate_pump(self.pump, t)
        if self.reservoir_components == 1:
            density = (field.real * field.real + field.imag * field.imag).sum(axis=1, keepdims=True)
            res_derivs = self.reservoir.get_derivatives(field[:, :1], pump, reservoir_fields, rho=density)
        else:
            res_derivs = self.reservoir.get_derivatives(field, pump * self.pump_weights, reservoir_fields)
        n_active = self.reservoir.get_active_density(reservoir_fields)
        n_inactive = self.reservoir.get_inactive_density(reservoir_fields)
        lap = out
        if lap is None:
            if self._laplacian_buf is None or self._laplacian_buf.shape != field.shape:
                self._laplacian_buf = xp.zeros_like(field)
            lap = self._laplacian_buf
        self._laplacian(field, lap)
        density = xp.abs(field) ** 2
        other_density = density[:, ::-1]
        zeeman_sign = xp.asarray([1, -1], dtype=density.dtype).reshape((1, 2, *(1,) * self.grid.ndim))
        kinetic = -(physics.get("hbar")**2) / (2 * physics.get("m_eff")) * lap
        eff_energy = (
            self.potential_at(t) + physics.get("g_C") * density
            + physics.get("alpha_2") * other_density
            + physics.get("zeeman") * zeeman_sign / 2
            + physics.get("g_R") * n_active + physics.get("g_I") * n_inactive
        )
        gain_loss = (physics.get("R") * n_active - physics.get("gamma_C")) / 2.0
        eta = physics.get("kinetic_relaxation_eta")
        hbar_over_2m = physics.get("hbar") / (2.0 * physics.get("m_eff"))
        dfield = (-1j / physics.get("hbar")) * (
            kinetic + eff_energy * field + self._te_tm_energy(field)
        ) + gain_loss * field + eta * n_active * hbar_over_2m * lap
        if self.terms:
            dfield = dfield + self.extra_rhs(field, reservoir_fields, t)
        if self._has_drive:
            dfield = dfield + self.drive_contribution(field, t)
        return dfield, res_derivs

    def _te_tm_energy(self, field):
        """Use central ∂xx−∂yy and a nine-point central ∂xy stencil.

        Periodic edges wrap. Closed-interval edges mirror the adjacent
        interior value, so the normal derivative and corner cross term vanish.
        """
        beta = self.physics.get("te_tm")
        if not self.physics.any_nonzero("te_tm"):
            return 0
        xp = self.xp
        other = field[:, ::-1]
        dx = self.grid.dx
        if self.grid.boundary == "periodic":
            dxx = (xp.roll(other, -1, axis=-1) - 2 * other + xp.roll(other, 1, axis=-1)) / dx**2
            if self.grid.ndim == 1:
                return -beta * dxx
            dy = self.grid.dy
            dyy = (xp.roll(other, -1, axis=-2) - 2 * other + xp.roll(other, 1, axis=-2)) / dy**2
            dxy = (xp.roll(xp.roll(other, -1, axis=-2), -1, axis=-1)
                   - xp.roll(xp.roll(other, -1, axis=-2), 1, axis=-1)
                   - xp.roll(xp.roll(other, 1, axis=-2), -1, axis=-1)
                   + xp.roll(xp.roll(other, 1, axis=-2), 1, axis=-1)) / (4 * dx * dy)
        else:
            padded = xp.pad(other, ((0, 0), (0, 0), *((1, 1),) * self.grid.ndim), mode="reflect")
            if self.grid.ndim == 1:
                dxx = (padded[..., 2:] - 2 * other + padded[..., :-2]) / dx**2
                return -beta * dxx
            dy = self.grid.dy
            dxx = (padded[..., 1:-1, 2:] - 2 * other + padded[..., 1:-1, :-2]) / dx**2
            dyy = (padded[..., 2:, 1:-1] - 2 * other + padded[..., :-2, 1:-1]) / dy**2
            dxy = (padded[..., 2:, 2:] - padded[..., 2:, :-2]
                   - padded[..., :-2, 2:] + padded[..., :-2, :-2]) / (4 * dx * dy)
        cross = xp.stack((-2j * dxy[:, 0], 2j * dxy[:, 1]), axis=1)
        return -beta * (dxx - dyy + cross)

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
