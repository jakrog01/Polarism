"""Device-side accumulated observables and their registry."""
from __future__ import annotations

from dataclasses import dataclass

from polarism.analysis import g1, g2
from polarism.compute_engine import compute_engine
from polarism.registry import Registry


@dataclass(frozen=True)
class AccumulatedResult:
    """One finalized accumulated array and its axis metadata."""
    name: str
    value: object
    dims: tuple[str, ...]
    units: str
    label: str = ""
    wigner_corrected: bool = False


available_accumulators: Registry[type] = Registry("accumulator")


def register_accumulator(name: str):
    """Register an accumulator class under a configuration name."""
    return available_accumulators.register(name)


class Accumulator:
    """Base class for scheduled device-side accumulators."""
    name = ""
    parameter_keys: frozenset[str] = frozenset()

    def __init__(self, controller, *, start=0.0, stop=None, every=None, **params):
        if params:
            raise ValueError(f"Unknown {self.name} parameters: {sorted(params)}")
        self.controller = controller
        self.xp = compute_engine.xp
        self.samples = 0
        self.t_first = None
        self.t_last = None
        self.start = start
        self.stop = stop
        self.every = every
        self.ndim = controller.grid.ndim
        self.dims = ((("member",) if controller.state.batched else ())
                     + (("component",) if controller.state.components == 2 else ())
                     + (("y", "x") if self.ndim == 2 else ("x",)))
        self.shape = controller.state.psi.shape
        self.wigner_corrected = controller.cfg.noise.enabled and controller.cfg.noise.stochastic == "wigner"
        cell = 1.0
        for spacing in controller.grid.spacing:
            cell *= spacing
        self.cell = cell
        self.cells = 1
        for size in controller.grid.shape:
            self.cells *= size
        members = controller._member_configs or [controller.cfg]
        strengths = [member.noise.stochastic_strength if member.noise.enabled and
                     member.noise.stochastic == "wigner" else 0.0 for member in members]
        self.delta = self.xp.asarray(strengths, dtype=self.xp.float64).reshape(
            ((len(strengths),) + (1,) * (len(self.shape) - 1)) if controller.state.batched
            else (1,) * len(self.shape)) / cell
        self.strength = self.delta * cell

    def update(self, state, t) -> None:
        """Accumulate one sample using device operations only."""
        raise NotImplementedError

    def _sampled(self, t):
        if self.samples == 0:
            self.t_first = t
        self.t_last = t
        self.samples += 1

    def state_dict(self) -> dict:
        """Return raw array sums and common sampling metadata."""
        return {"samples": self.samples, "t_first": self.t_first, "t_last": self.t_last}

    def load_state_dict(self, values) -> None:
        """Restore common sampling metadata."""
        self.samples = int(values["samples"])
        self.t_first = values["t_first"]
        self.t_last = values["t_last"]

    def results(self) -> list[AccumulatedResult]:
        """Return normalized results after the run."""
        raise NotImplementedError


@register_accumulator("momentum-distribution")
class MomentumDistribution(Accumulator):
    """Accumulate the per-mode particle number and first-order correlation."""
    name = "momentum-distribution"

    def __init__(self, controller, **params):
        super().__init__(controller, **params)
        self.sum_k = self.xp.zeros(self.shape, dtype=self.xp.float64)

    def update(self, state, t) -> None:
        axes = tuple(range(-self.ndim, 0))
        field_k = self.xp.fft.fftn(state.field, axes=axes)
        public = field_k if state.batched else field_k[0]
        if state.components == 1:
            public = public[:, 0] if state.batched else public[0]
        real = public.real.astype(self.xp.float64)
        imag = public.imag.astype(self.xp.float64)
        self.sum_k += real * real + imag * imag
        self._sampled(t)

    def state_dict(self):
        return {**super().state_dict(), "sum_k": self.sum_k}

    def load_state_dict(self, values):
        super().load_state_dict(values)
        self.sum_k = self.xp.asarray(values["sum_k"], dtype=self.xp.float64)

    def results(self):
        axes = tuple(range(-self.ndim, 0))
        value = self.xp.fft.fftshift(self.sum_k * (self.cell / (self.cells * self.samples)), axes=axes)
        if self.wigner_corrected:
            value = value - self.strength / 2
        dims = tuple("k" + axis if axis in ("x", "y") else axis for axis in self.dims)
        return [AccumulatedResult("momentum_distribution", value, dims, "particles/mode",
                                  "Momentum distribution", self.wigner_corrected),
                AccumulatedResult("g1", g1(value, ndim=self.ndim), self.dims, "",
                                  "First-order spatial correlation", self.wigner_corrected)]


@register_accumulator("g2")
class G2Accumulator(Accumulator):
    """Accumulate normally ordered density moments and their ratio."""
    name = "g2"

    def __init__(self, controller, **params):
        super().__init__(controller, **params)
        self.sum_density = self.xp.zeros(self.shape, dtype=self.xp.float64)
        self.sum_density_squared = self.xp.zeros(self.shape, dtype=self.xp.float64)

    def update(self, state, t) -> None:
        real = state.psi.real.astype(self.xp.float64)
        imag = state.psi.imag.astype(self.xp.float64)
        density = real * real + imag * imag
        self.sum_density += density
        self.sum_density_squared += density * density
        self._sampled(t)

    def state_dict(self):
        return {**super().state_dict(), "sum_density": self.sum_density,
                "sum_density_squared": self.sum_density_squared}

    def load_state_dict(self, values):
        super().load_state_dict(values)
        self.sum_density = self.xp.asarray(values["sum_density"], dtype=self.xp.float64)
        self.sum_density_squared = self.xp.asarray(values["sum_density_squared"], dtype=self.xp.float64)

    def results(self):
        mean = self.sum_density / self.samples
        squared = self.sum_density_squared / self.samples
        if self.wigner_corrected:
            squared = squared - 2 * self.delta * mean + self.delta * self.delta / 2
            mean = mean - self.delta / 2
        units = f"µm^-{self.ndim}"
        return [AccumulatedResult("density_mean", mean, self.dims, units,
                                  "Mean density", self.wigner_corrected),
                AccumulatedResult("density_squared_mean", squared, self.dims, f"µm^-{2 * self.ndim}",
                                  "Mean squared density", self.wigner_corrected),
                AccumulatedResult("g2", g2(mean, squared), self.dims, "",
                                  "Second-order correlation", self.wigner_corrected)]
