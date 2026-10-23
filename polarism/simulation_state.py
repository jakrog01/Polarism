"""Simulation state storage."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.init_condition import make_initial_psi

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.grid.simulation_grid import SimulationGrid


class SimulationState:
    """Store canonical fields and expose scalar public wavefunction views."""
    _field: Union[np.ndarray, cp.ndarray] | None
    _field_k: Union[np.ndarray, cp.ndarray] | None
    t: float

    def __init__(
        self,
        grid: SimulationGrid,
        eps: float = 1e-3,
        precision: str = "double",
        *,
        mode: str = "legacy_positive_uniform",
        k_cutoff_um: Optional[float] = None,
        seed: Optional[int] = None,
        batch_size: int | None = None,
    ):
        """Initialize each member from a seeded spatial field."""
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = 1 if batch_size is None else batch_size
        self.components = 1
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
        if self.batched:
            psi = xp.broadcast_to(psi, (self.batch_size, *grid.shape)).copy()
        self._field = psi.reshape(self.batch_size, self.components, *grid.shape)
        self._field_k = None
        self.t = 0.0

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
        return ((self.batch_size, *self._spatial_shape) if self.batched
                else self._spatial_shape)

    @property
    def psi(self) -> Union[np.ndarray, cp.ndarray]:
        """Return a public-shape view of the real-space wavefunction."""
        field = self.field
        return field[:, 0] if self.batched else field[0, 0]

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
