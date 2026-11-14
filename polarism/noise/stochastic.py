"""Apply a counter-based truncated-Wigner increment after each solver step."""
from __future__ import annotations

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.noise.philox import counter_words, cuda_source, normal_pair, philox4x32


def draw_seed() -> int:
    """Resolve a non-negative int64 seed from operating-system entropy."""
    return int(np.random.SeedSequence().entropy % (1 << 63))


class NoiseApplier:
    """Hold immutable per-member parameters and apply stateless increments."""

    def __init__(self, config, grid, members=None):
        self.xp = compute_engine.xp
        self.members = members or [config]
        self.precision = config.solver.precision
        self.spatial_n = int(np.prod(grid.shape))
        self.volume = float(np.prod(grid.spacing))
        self.dt = float(config.solver.dt)
        self.seeds = np.asarray([member.noise.seed for member in self.members], dtype=np.uint64)
        self.gamma = np.asarray([member.physics.gamma_C for member in self.members], dtype=np.float64)
        self.scattering = np.asarray([member.physics.R for member in self.members], dtype=np.float64)
        self.strength = np.asarray([member.noise.stochastic_strength for member in self.members], dtype=np.float64)
        self.active = bool(config.noise.enabled and config.noise.stochastic == "wigner"
                           and np.any(self.strength > 0.0))
        if self.active and compute_engine.use_gpu:
            dtype = self.xp.float32 if self.precision == "single" else self.xp.float64
            self._device_seeds = self.xp.asarray(self.seeds)
            self._device_gamma = self.xp.asarray(self.gamma, dtype=dtype)
            self._device_scattering = self.xp.asarray(self.scattering, dtype=dtype)
            self._device_strength = self.xp.asarray(self.strength, dtype=dtype)
            suffix = self.precision
            self._kernel = self.xp.RawKernel(cuda_source(suffix), f"add_wigner_{suffix}")

    def apply_arrays(self, field, active_density, step: int):
        """Add one increment to a canonical real-space field."""
        if not self.active:
            return
        if compute_engine.use_gpu:
            total_n = int(field.size)
            dtype = self.xp.float32 if self.precision == "single" else self.xp.float64
            if self.xp.ndim(active_density) == 0:
                active_density = self.xp.zeros(field.shape, dtype=dtype)
            self._kernel(((total_n + 255) // 256,), (256,), (
                field.view(dtype).ravel(), active_density.ravel(),
                self._device_seeds, self._device_gamma, self._device_scattering,
                self._device_strength, np.uint64(step), self.spatial_n, total_n,
                dtype(self.dt / (4.0 * self.volume)),
            ))
            return
        real_dtype = np.float32 if self.precision == "single" else np.float64
        for member in range(field.shape[0]):
            if self.strength[member] == 0.0:
                continue
            words = philox4x32(counter_words(np.arange(self.spatial_n), step),
                               int(self.seeds[member]))
            re, im = normal_pair(words, self.precision)
            density = (0 if np.ndim(active_density) == 0 else
                       np.asarray(active_density[member, 0]).reshape(-1))
            sigma = np.sqrt(real_dtype(self.strength[member]) *
                            (real_dtype(self.gamma[member])
                             + real_dtype(self.scattering[member]) * density)
                            * real_dtype(self.dt / (4.0 * self.volume)))
            field[member, 0] += (sigma * (re + 1j * im)).reshape(field.shape[2:])

    def apply_to_state(self, state, reservoir, step: int):
        """Update the field and clear any stale Fourier cache."""
        if not self.active:
            return
        field = state.field
        active = reservoir.get_active_density(reservoir.get_state())
        self.apply_arrays(field, active, step)
        state.invalidate_psi_k()
