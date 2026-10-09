"""Quadratic two-reservoir model.

State variables
---------------
nR : active reservoir density
    Fed by quadratic transfer from the inactive reservoir:
    ``kappa * nI²``.  Drives stimulated scattering into the condensate.
nI : inactive reservoir density
    Directly fed by the external pump ``P(x,y,t)``.

Equations of motion (state order: ``(nR, nI)``)
-------------------------------------------------
``dnR/dt = kappa * nI² - gammaR * nR - R * nR * |ψ|²``
``dnI/dt = P(x,y,t) - kappa * nI² - gammaI * nI``

The active density ``nR`` is passed to the GPE as the reservoir source term.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import real_dtype
from polarism.config.simulation_parameters import PhysicsConstants, ReservoirParameters
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.laplacian import real_laplacian
from polarism.results.result_node import ResultNode
from polarism.results.result_provider import ResultProvider

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np


_QDOUBLE_RHS_KERNELS: dict[tuple[str, int], object] = {}


class QuadraticDoubleReservoir(AbstractReservoir, ResultProvider):
    """Two-reservoir model with quadratic inactive→active transfer."""

    config: ReservoirParameters
    R: float
    gamma_R: float
    gamma_I: float
    kappa: float
    reservoir_diffusion_I: float
    reservoir_diffusion_R: float
    dx: float
    dy: float
    grid_type: str
    nR: Union["np.ndarray", "cp.ndarray"]
    nI: Union["np.ndarray", "cp.ndarray"]

    def __init__(
        self,
        reservoir_config: ReservoirParameters,
        physics: PhysicsConstants,
        grid: SimulationGrid2D,
        precision: str = "double",
    ) -> None:
        """Set up the quadratic double reservoir."""
        self.xp = compute_engine.xp
        self.config = reservoir_config
        self.R = physics.R
        self.gamma_R = physics.gamma_R
        self.gamma_I = physics.gamma_I
        self.kappa = physics.kappa
        self.reservoir_diffusion_I = getattr(physics, "reservoir_diffusion_I", 0.0)
        self.reservoir_diffusion_R = getattr(physics, "reservoir_diffusion_R", 0.0)
        self.dx = grid.dx
        self.dy = grid.dy
        self.grid_type = getattr(grid, "grid_type", "periodic")
        rdtype = real_dtype(self.xp, precision)
        self.nR = self.xp.zeros((grid.ny, grid.nx), dtype=rdtype)
        self.nI = self.xp.zeros((grid.ny, grid.nx), dtype=rdtype)

    def get_state(
        self,
    ) -> tuple[Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]]:
        """Return the current reservoir state as ``(nR, nI)``."""
        return (self.nR, self.nI)

    def set_state(
        self,
        state: tuple[
            Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]
        ],
    ) -> None:
        """Set the reservoir state from ``(nR, nI)``."""
        self.nR = self.xp.maximum(state[0], 0).astype(self.nR.dtype, copy=False)
        self.nI = self.xp.maximum(state[1], 0).astype(self.nI.dtype, copy=False)

    def get_active_density(
        self,
        state: tuple[
            Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]
        ],
    ) -> Union["np.ndarray", "cp.ndarray"]:
        """Return the active reservoir density ``nR``."""
        return state[0]

    def get_inactive_density(
        self,
        state: tuple[
            Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]
        ],
    ) -> Union["np.ndarray", "cp.ndarray"]:
        """Return the inactive reservoir density ``nI``."""
        return state[1]

    def get_derivatives(
        self,
        psi: Union["np.ndarray", "cp.ndarray"],
        Pxy: Union["np.ndarray", "cp.ndarray"],
        state: tuple[
            Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]
        ],
        rho: Union["np.ndarray", "cp.ndarray"] | None = None,
    ) -> tuple[
        Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]
    ]:
        """Return time derivatives ``(dnR, dnI)``."""
        nR, nI = state
        abs_psi2 = rho if rho is not None else psi.real * psi.real + psi.imag * psi.imag

        if nR.shape[-2:] != psi.shape[-2:] or nI.shape[-2:] != psi.shape[-2:]:
            raise ValueError(
                "reservoir state and psi must share trailing grid shape, "
                f"got nR={nR.shape}, nI={nI.shape}, psi={psi.shape}"
            )
        if Pxy.shape[-2:] != psi.shape[-2:]:
            raise ValueError(
                "pump and psi must share trailing grid shape, "
                f"got pump={Pxy.shape}, psi={psi.shape}"
            )

        kernel = self._get_qdouble_rhs_kernel()
        if kernel is None:
            transfer = self.kappa * nI * nI
            dnR = transfer - self.gamma_R * nR - self.R * nR * abs_psi2
            dnI = Pxy - transfer - self.gamma_I * nI
        else:
            if abs_psi2.dtype != self.nR.dtype:
                raise TypeError(
                    "rho and reservoir dtypes must match for qdouble_rhs, "
                    f"got rho={abs_psi2.dtype}, reservoir={self.nR.dtype}"
                )
            dnR, dnI = kernel(
                nR,
                nI,
                abs_psi2,
                Pxy,
                self.kappa,
                self.gamma_R,
                self.gamma_I,
                self.R,
            )

        if self.reservoir_diffusion_R != 0.0:
            dnR = dnR + self.reservoir_diffusion_R * real_laplacian(
                nR, self.xp, self.dx, self.dy, self.grid_type
            )
        if self.reservoir_diffusion_I != 0.0:
            dnI = dnI + self.reservoir_diffusion_I * real_laplacian(
                nI, self.xp, self.dx, self.dy, self.grid_type
            )

        return (dnR, dnI)

    def _get_qdouble_rhs_kernel(self):
        if getattr(self.xp, "__name__", "") != "cupy":
            return None
        dtype_name = self.nR.dtype.name
        device_id = int(self.xp.cuda.runtime.getDevice())
        cache_key = (dtype_name, device_id)
        kernel = _QDOUBLE_RHS_KERNELS.get(cache_key)
        if kernel is not None:
            return kernel
        kernel = self.xp.ElementwiseKernel(
            "T nR, T nI, T rho, T Pxy, T kappa, T gammaR, T gammaI, T R",
            "T dnR, T dnI",
            """
            T transfer = kappa * nI * nI;
            dnR = transfer - gammaR * nR - R * nR * rho;
            dnI = Pxy - transfer - gammaI * nI;
            """,
            f"qdouble_rhs_{dtype_name}",
        )
        _QDOUBLE_RHS_KERNELS[cache_key] = kernel
        return kernel

    def step(
        self,
        dt: float,
        psi: Union["np.ndarray", "cp.ndarray"],
        Pxy: Union["np.ndarray", "cp.ndarray"],
    ) -> None:
        """Advance the reservoir by one time step with midpoint RK2."""
        s0 = self.get_state()
        k1 = self.get_derivatives(psi, Pxy, s0)
        s_mid = (s0[0] + 0.5 * dt * k1[0], s0[1] + 0.5 * dt * k1[1])
        k2 = self.get_derivatives(psi, Pxy, s_mid)
        self.set_state((s0[0] + dt * k2[0], s0[1] + dt * k2[1]))

    def get_reservoir_density(self) -> Union["np.ndarray", "cp.ndarray"]:
        """Return the active reservoir density ``nR``."""
        return self.nR

    def get_reservoir_densities(
        self,
    ) -> tuple[Union["np.ndarray", "cp.ndarray"], Union["np.ndarray", "cp.ndarray"]]:
        """Return ``(nR, nI)``."""
        return self.nR, self.nI

    def make_result_nodes(self) -> list[ResultNode]:
        """Build result nodes exposed by this reservoir."""
        nodes: list[ResultNode] = []
        if self.config.expose_results:
            nodes.append(
                ResultNode(
                    "nR",
                    compute_fn=lambda **ctx: self.nR,
                    reduce_dim_fn=lambda f: float(f.max()),
                    cmap="viridis",
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=True,
                )
            )
            nodes.append(
                ResultNode(
                    "nI",
                    compute_fn=lambda **ctx: self.nI,
                    reduce_dim_fn=lambda f: float(f.max()),
                    cmap="plasma",
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=True,
                )
            )
            nodes.append(
                ResultNode(
                    "nR_max",
                    compute_fn=lambda **ctx: float(self.nR.max()),
                    reduce_dim_fn=lambda v: float(v),
                    cmap=None,
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=False,
                )
            )
            nodes.append(
                ResultNode(
                    "nI_max",
                    compute_fn=lambda **ctx: float(self.nI.max()),
                    reduce_dim_fn=lambda v: float(v),
                    cmap=None,
                    scaling=None,
                    clim=None,
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=False,
                )
            )
        return nodes
