"""Open registry of dispersion relations for scalar and spinor fields."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

from polarism.registry import Registry
from polarism.compute_engine import compute_engine

available_dispersions: Registry[type["Dispersion"]] = Registry("dispersion")


def register_dispersion(name: str) -> Callable[[type["Dispersion"]], type["Dispersion"]]:
    """Return a decorator that registers a dispersion and sets its ``kind``."""
    def decorator(component):
        """Register ``component`` and record its name."""
        component.kind = name
        return available_dispersions.register(name)(component)

    return decorator


class Dispersion(ABC):
    """Provide the kinetic energy E(k) on the FFT grid."""

    kind: ClassVar[str]

    @abstractmethod
    def energy(self, grid, physics):
        """Return real E(k) in meV, broadcastable to the field shape."""
        raise NotImplementedError

    def matrix(self, grid, physics, components):
        """Return the spectral Hamiltonian with axes ``(batch, C, C, *space)``."""
        if components not in (1, 2):
            raise ValueError("components must be 1 or 2")
        xp = compute_engine.xp
        energy = self.energy(grid, physics)
        energy = xp.asarray(energy)
        if energy.ndim == 0:
            energy = xp.broadcast_to(energy, grid.shape)[None]
        elif energy.ndim == grid.ndim + 2:
            energy = energy[:, 0]
        elif energy.ndim == grid.ndim:
            energy = energy[None]
        energy = xp.broadcast_to(energy, (energy.shape[0], *grid.shape))
        if components == 1:
            return energy[:, None, None]

        zeeman = physics.get("zeeman") / 2
        beta = physics.get("te_tm")
        if hasattr(zeeman, "ndim") and zeeman.ndim:
            zeeman = zeeman[:, 0]
        if hasattr(beta, "ndim") and beta.ndim:
            beta = beta[:, 0]
        ky = grid.KY if grid.ndim == 2 else 0
        coupling = beta * (grid.KX - 1j * ky) ** 2
        upper, lower, diagonal_plus, diagonal_minus = xp.broadcast_arrays(
            coupling, xp.conj(coupling), energy + zeeman, energy - zeeman
        )
        return xp.stack((xp.stack((diagonal_plus, upper), axis=1),
                         xp.stack((lower, diagonal_minus), axis=1)), axis=1)


def spinor_matrix_exponential(matrix, hbar, dt, xp):
    """Return ``exp(-i H dt / hbar)`` from the traceless 2×2 identity."""
    if hasattr(hbar, "ndim") and hbar.ndim:
        hbar = hbar[:, 0]
    epsilon = (matrix[:, 0, 0] + matrix[:, 1, 1]) / 2
    z = (matrix[:, 0, 0] - matrix[:, 1, 1]) / 2
    upper = matrix[:, 0, 1]
    lower = matrix[:, 1, 0]
    omega = xp.sqrt(z * z + upper * lower)
    tau = dt / hbar
    theta = omega * tau
    phase = xp.exp(-1j * epsilon * tau)
    cosine = xp.cos(theta)
    sine_over_omega = tau * xp.sinc(theta / xp.pi)
    diagonal_plus = phase * (cosine - 1j * sine_over_omega * z)
    diagonal_minus = phase * (cosine + 1j * sine_over_omega * z)
    return xp.stack((xp.stack((diagonal_plus, -1j * phase * sine_over_omega * upper), axis=1),
                     xp.stack((-1j * phase * sine_over_omega * lower, diagonal_minus), axis=1)), axis=1)


def apply_spinor_matrix(matrix, field, xp):
    """Apply a precomputed spectral matrix along the component axis."""
    return xp.stack((matrix[:, 0, 0] * field[:, 0] + matrix[:, 0, 1] * field[:, 1],
                     matrix[:, 1, 0] * field[:, 0] + matrix[:, 1, 1] * field[:, 1]), axis=1)


@register_dispersion("parabolic")
class ParabolicDispersion(Dispersion):
    """Return ``hbar^2 k^2 / (2 m_eff)``."""

    def energy(self, grid, physics):
        """Return the parabolic kinetic energy."""
        return physics.get("hbar") ** 2 * grid.k_squared / (2 * physics.get("m_eff"))
