"""Device-side accumulated observables and their registry."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

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
    linear: bool = False


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
                                  "Momentum distribution", self.wigner_corrected, True),
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
                                  "Mean density", self.wigner_corrected, True),
                AccumulatedResult("density_squared_mean", squared, self.dims, f"µm^-{2 * self.ndim}",
                                  "Mean squared density", self.wigner_corrected, True),
                AccumulatedResult("g2", g2(mean, squared), self.dims, "",
                                  "Second-order correlation", self.wigner_corrected)]


class Spectrum(Accumulator):
    """Accumulate a time-resolved field cut using overlapping Hann windows."""
    parameter_keys = frozenset({"segment", "y"})
    axis = ""

    def __init__(self, controller, *, segment=1024, y=0.0, **params):
        super().__init__(controller, **params)
        self.segment = segment
        self.y = y
        self.position = 0
        self.since_segment = 0
        self.segments = 0
        self.row = None
        if self.ndim == 2 and self.axis == "x":
            self.row = int(self.xp.argmin(self.xp.abs(controller.grid.Y[:, 0] - y)))
        cut_shape = (*self.shape[:-2], self.shape[-1]) if self.ndim == 2 else self.shape
        self.ring = self.xp.zeros((segment, *cut_shape), dtype=controller.state.field.dtype)
        self.psd_sum = self.xp.zeros((*cut_shape[:-1], segment, cut_shape[-1]), dtype=self.xp.float64)
        indices = self.xp.arange(segment, dtype=self.xp.float64)
        self.window = 0.5 - 0.5 * self.xp.cos(2 * self.xp.pi * indices / segment)
        self.window = self.window.reshape((segment,) + (1,) * len(cut_shape))

    def update(self, state, t) -> None:
        if self.axis == "k":
            line = state.field.sum(axis=-2) if self.ndim == 2 else state.field
            line = self.xp.fft.fft(line, axis=-1)
            line = line * (self.cell / self.cells) ** 0.5
            line = line if state.batched else line[0]
            if state.components == 1:
                line = line[:, 0] if state.batched else line[0]
            line = self.xp.fft.fftshift(line, axes=-1)
        else:
            line = state.psi[..., self.row, :] if self.ndim == 2 else state.psi
        self.ring[self.position] = line
        self.position = (self.position + 1) % self.segment
        self.since_segment += 1
        self._sampled(t)
        if self.samples >= self.segment and (self.segments == 0 or self.since_segment >= self.segment // 2):
            block = self.xp.concatenate((self.ring[self.position:], self.ring[:self.position]), axis=0)
            transformed = self.xp.fft.fft(block * self.window, axis=0)
            real = transformed.real.astype(self.xp.float64)
            imag = transformed.imag.astype(self.xp.float64)
            power = self.xp.moveaxis(real * real + imag * imag, 0, -2)
            self.psd_sum += power
            self.segments += 1
            self.since_segment = 0

    def state_dict(self):
        return {**super().state_dict(), "psd_sum": self.psd_sum, "ring": self.ring,
                "position": self.xp.asarray([self.position]),
                "since_segment": self.xp.asarray([self.since_segment]),
                "segments": self.xp.asarray([self.segments])}

    def load_state_dict(self, values):
        super().load_state_dict(values)
        self.psd_sum = self.xp.asarray(values["psd_sum"], dtype=self.xp.float64)
        self.ring = self.xp.asarray(values["ring"], dtype=self.ring.dtype)
        self.position = int(values["position"][0])
        self.since_segment = int(values["since_segment"][0])
        self.segments = int(values["segments"][0])

    def results(self):
        if not self.segments:
            warnings.warn(f"{self.name} has no completed segments; no spectrum emitted", UserWarning, stacklevel=2)
            return []
        hbar = self.controller.cfg.physics.hbar
        spacing = self.every or self.controller.cfg.result.save_every or self.controller.cfg.solver.dt
        frequencies = self.xp.fft.fftfreq(self.segment, d=spacing)
        order = self.xp.argsort(-frequencies)
        delta_energy = 2 * self.xp.pi * hbar / (self.segment * spacing)
        energy = -2 * self.xp.pi * hbar * frequencies[order]
        members = self.controller._member_configs or [self.controller.cfg]
        shifts = [member.drive.energy if member.drive.enabled else 0.0 for member in members]
        if self.controller.state.batched:
            energy = energy[None, :] + self.xp.asarray(shifts)[:, None]
        else:
            energy = energy + shifts[0]
        window_power = self.xp.sum(self.window[:, 0] ** 2) if self.window.ndim > 1 else self.xp.sum(self.window ** 2)
        value = self.xp.take(self.psd_sum, order, axis=-2)
        value = value / (self.segments * self.segment * window_power * delta_energy)
        prefix = (("member",) if self.controller.state.batched else ()) + (("component",) if self.controller.state.components == 2 else ())
        name = f"spectrum_{self.axis}"
        units = "particles/mode/meV" if self.axis == "k" else f"µm^-{self.ndim} meV^-1"
        return [AccumulatedResult(name, value, (*prefix, "E", "kx" if self.axis == "k" else "x"),
                                  units, f"Spectrum {self.axis}", False, True),
                AccumulatedResult(f"{name}_energy", energy,
                                  (("member",) if self.controller.state.batched else ()) + ("E",),
                                  "meV", "Energy")]


@register_accumulator("spectrum-k")
class SpectrumK(Spectrum):
    """Momentum cut at ky = 0."""
    name = "spectrum-k"
    axis = "k"
    parameter_keys = frozenset({"segment"})


@register_accumulator("spectrum-x")
class SpectrumX(Spectrum):
    """Real-space cut at the row nearest y."""
    name = "spectrum-x"
    axis = "x"
