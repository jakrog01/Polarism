"""A one-field reservoir implemented directly from AbstractReservoir."""

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.reservoir import AbstractReservoir, register_reservoir
from polarism.reservoir.laplacian import real_laplacian


@register_reservoir("example-slow-single")
class SlowSingleReservoir(AbstractReservoir):
    def __init__(self, reservoir_config, physics, grid, precision="double"):
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.gamma_r = physics.gamma_R * reservoir_config.extra.get("decay_factor", 1.0)
        self.R = physics.R
        self.diffusion = physics.reservoir_diffusion_R
        self.dx = grid.dx
        self.dy = grid.dy if grid.ndim == 2 else None
        self.boundary = grid.boundary
        self.nR = self.xp.zeros((1, 1, *grid.shape), dtype=real_dtype(self.xp, precision))

    def get_state(self):
        return (self.nR,)

    def set_state(self, state):
        self.nR = self.xp.maximum(state[0], 0).astype(self.nR.dtype, copy=False)

    def get_reservoir_density(self):
        return self.nR[0, 0]

    def get_derivatives(self, psi, pump, state, rho=None):
        density = state[0]
        if rho is None:
            rho = psi.real * psi.real + psi.imag * psi.imag
        rate = pump - (self.gamma_r + self.R * rho) * density
        if self.diffusion != 0:
            rate += self.diffusion * real_laplacian(
                density, self.xp, self.dx, self.dy, self.boundary
            )
        return (rate,)

    def step(self, dt, psi_mid, pump):
        fields0 = self.get_state()
        (k1,) = self.get_derivatives(psi_mid, pump, fields0)
        midpoint = (fields0[0] + 0.5 * dt * k1,)
        (k2,) = self.get_derivatives(psi_mid, pump, midpoint)
        self.set_state((fields0[0] + dt * k2,))

    def make_result_nodes(self):
        return []
