"""Simulation state storage."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.noise.initial_field import make_initial_psi
from polarism.initial_condition import create_initial_condition

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import InitialConditionParameters

    from polarism.grid.simulation_grid import SimulationGrid


class SimulationState:
    """Store canonical fields and expose scalar public wavefunction views."""
    _field: Union[np.ndarray, cp.ndarray] | None
    _field_k: Union[np.ndarray, cp.ndarray] | None
    t: float
    t0: float
    step: int

    def __init__(
        self,
        grid: SimulationGrid,
        eps: float = 1e-3,
        precision: str = "double",
        *,
        mode: str = "complex_gaussian_zero_mean",
        k_cutoff_um: Optional[float] = None,
        seed: Optional[int] = None,
        batch_size: int | None = None,
        initial_condition: InitialConditionParameters | None = None,
        components: int = 1,
    ):
        """Initialize each member from a seeded spatial field.

        A non-``zero`` *initial_condition* is added to the noise field.
        """
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = 1 if batch_size is None else batch_size
        if components not in (1, 2):
            raise ValueError("components must be 1 or 2")
        self.components = components
        self.batched = batch_size is not None
        self._spatial_shape = grid.shape
        self._fft_axes = tuple(range(-grid.ndim, 0))
        xp = compute_engine.xp
        rdtype = real_dtype(xp, precision)
        cdtype = complex_dtype(xp, precision)
        psi = make_initial_psi(
            xp, grid.shape, grid.spacing, eps,
            mode=mode,
            k_cutoff_um=k_cutoff_um,
            seed=seed,
            cdtype=cdtype,
            rdtype=rdtype,
        )
        if components == 2:
            other = make_initial_psi(
                xp, grid.shape, grid.spacing, eps,
                mode=mode, k_cutoff_um=k_cutoff_um,
                seed=None if seed is None else seed + 1,
                cdtype=cdtype, rdtype=rdtype,
            )
            psi = xp.stack((psi, other))
        if initial_condition is not None:
            deterministic = create_initial_condition(grid, initial_condition, xp, cdtype)
            if deterministic is not None:
                if components == 1 and deterministic.shape != grid.shape:
                    raise ValueError("scalar state requires a scalar initial condition")
                if components == 2 and deterministic.shape == grid.shape:
                    psi[0] = psi[0] + deterministic
                else:
                    psi = psi + deterministic
        if self.batched:
            psi = xp.broadcast_to(psi, (self.batch_size, *((components,) if components == 2 else ()), *grid.shape)).copy()
        self._field = psi.reshape(self.batch_size, self.components, *grid.shape)
        self._field_k = None
        self.t = 0.0
        self.t0 = 0.0
        self.step = 0

    @property
    def field(self) -> Union[np.ndarray, cp.ndarray]:
        """Return the canonical real-space field, materializing it if needed."""
        if self._field is None:
            self.materialize_psi(compute_engine.xp)
        if self._field is None:
            raise ValueError("SimulationState.field is unavailable")
        return self._field

    @field.setter
    def field(self, value: Union[np.ndarray, cp.ndarray] | None) -> None:
        if value is not None and value.shape != self._canonical_shape:
            raise ValueError(f"field shape must be {self._canonical_shape}, got {value.shape}")
        self._field = value
        self._field_k = None

    @property
    def field_k(self) -> Union[np.ndarray, cp.ndarray] | None:
        """Return the cached canonical spectral field."""
        return self._field_k

    @field_k.setter
    def field_k(self, value: Union[np.ndarray, cp.ndarray] | None) -> None:
        if value is not None and value.shape != self._canonical_shape:
            raise ValueError(f"field_k shape must be {self._canonical_shape}, got {value.shape}")
        self._field_k = value

    @property
    def _canonical_shape(self) -> tuple[int, ...]:
        return (self.batch_size, self.components, *self._spatial_shape)

    @property
    def _public_shape(self) -> tuple[int, ...]:
        if self.components == 2:
            return ((self.batch_size, 2, *self._spatial_shape) if self.batched
                    else (2, *self._spatial_shape))
        return ((self.batch_size, *self._spatial_shape) if self.batched
                else self._spatial_shape)

    @property
    def psi(self) -> Union[np.ndarray, cp.ndarray]:
        """Return a public-shape view of the real-space wavefunction."""
        field = self.field
        if self.components == 2:
            return field if self.batched else field[0]
        return field[:, 0] if self.batched else field[0, 0]

    @property
    def psi_plus(self):
        """Return the σ+ circular component."""
        if self.components != 2:
            raise AttributeError("psi_plus requires a spinor state")
        return self.psi[:, 0] if self.batched else self.psi[0]

    @property
    def psi_minus(self):
        """Return the σ− circular component."""
        if self.components != 2:
            raise AttributeError("psi_minus requires a spinor state")
        return self.psi[:, 1] if self.batched else self.psi[1]

    @psi.setter
    def psi(self, value: Union[np.ndarray, cp.ndarray] | None) -> None:
        if value is None:
            self.field = None
            return
        if value.shape != self._public_shape:
            raise ValueError(f"psi shape must be {self._public_shape}, got {value.shape}")
        self.field = value.reshape(self._canonical_shape)

    @property
    def psi_k(self) -> Union[np.ndarray, cp.ndarray] | None:
        """Return a public-shape view of the cached spectral field."""
        field_k = self.field_k
        if field_k is None:
            return None
        if self.components == 2:
            return field_k if self.batched else field_k[0]
        return field_k[:, 0] if self.batched else field_k[0, 0]

    @psi_k.setter
    def psi_k(self, value: Union[np.ndarray, cp.ndarray] | None) -> None:
        if value is None:
            self.field_k = None
            return
        if value.shape != self._public_shape:
            raise ValueError(f"psi_k shape must be {self._public_shape}, got {value.shape}")
        self.field_k = value.reshape(self._canonical_shape)

    def materialize_psi(self, xp) -> None:
        """Materialize the canonical real-space field from its spectral cache."""
        if self._field is None:
            if self._field_k is None:
                raise ValueError("Cannot materialize psi without psi_k")
            self._field = (xp.fft.ifft2(self._field_k, axes=self._fft_axes)
                           if len(self._spatial_shape) == 2
                           else xp.fft.ifft(self._field_k, axis=-1))

    def invalidate_psi_k(self) -> None:
        """Clear the spectral cache after external real-space mutation."""
        self.field_k = None

    def get_simulation_state(self) -> tuple[Union[np.ndarray, cp.ndarray], float]:
        """Return the public wavefunction and current time."""
        return self.psi, self.t
