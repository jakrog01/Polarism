"""Two-transform CUDA Strang solver with local reservoir and condensate steps."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import re

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.config.batch_physics import BatchPhysics
from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


# The same source is also compiled directly by the CPU-only NVRTC smoke test.
# Reservoir state order is (active, inactive) for both two-field models.
_LOCAL_KERNEL = r"""
typedef {real} real;
#define MODEL {model}
__device__ __forceinline__ real pump_value(const real* p, real scale, int j) {{
    return {mul}(scale, p[j]);
}}
__device__ __forceinline__ void rhs(real a, real b, real rho, real p,
    real R, real gammaR, real gammaI, real gammaA, real ria, real rai,
    real kappa, real* da, real* db) {{
#if MODEL == 0
    *da = p - (gammaR + R*rho)*a;
    *db = 0;
#elif MODEL == 1
    *da = ria*b - (gammaA + rai + R*rho)*a;
    *db = p - (gammaI + ria)*b + rai*a;
#else
    real transfer = kappa*b*b;
    *da = transfer - gammaR*a - R*a*rho;
    *db = p - transfer - gammaI*b;
#endif
}}
__device__ __forceinline__ void half_step(real* a, real* b, real rho,
    real p, real h, real R, real gammaR, real gammaI, real gammaA,
    real ria, real rai, real kappa) {{
    real da1, db1, da2, db2;
    rhs(*a, *b, rho, p, R, gammaR, gammaI, gammaA, ria, rai, kappa, &da1, &db1);
    rhs(*a + (real)0.5*h*da1, *b + (real)0.5*h*db1, rho, p,
        R, gammaR, gammaI, gammaA, ria, rai, kappa, &da2, &db2);
    *a = fmax(*a + h*da2, (real)0);
#if MODEL != 0
    *b = fmax(*b + h*db2, (real)0);
#endif
}}
__device__ __forceinline__ void nonlinear(real* re, real* im,
    real a, real b, real vr, real vi, real vi_scale, real dt,
    real inv_hbar, real gC, real gR, real gI, real R, real gammaC) {{
    real rho = (*re)*(*re) + (*im)*(*im);
    real gain = (R*a - gammaC)*(real)0.5 + vi_scale*vi*inv_hbar;
    real x = (real)2*gain*dt;
    real phase_time = fabs(x) < (real)1e-6
        ? dt*((real)1 + x*(real)0.5 + x*x/(real)6)
        : dt*expm1(x)/x;
    real phase = ((vr + gR*a + gI*b)*dt + gC*rho*phase_time)*inv_hbar;
    real amp = exp(gain*dt);
    real c, s;
    sincos(phase, &s, &c);
    real r = *re, i = *im;
    *re = amp*(r*c + i*s);
    *im = amp*(i*c - r*s);
}}
extern "C" __global__ void local_step(
    const real* psi, const real* a0, const real* b0,
    const real* vr, const real* vi, const real* p1, const real* p2,
    real* out_psi, real* out_a, real* out_b,
    int n, int spatial_n, int p1_n, int p2_n, real scale1, real scale2, real vi_scale, real dt,
    real inv_hbar, real gC, real gR, real gI, real R, real gammaC,
    real gammaR, real gammaI, real gammaA, real ria, real rai, real kappa) {{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n) return;
    real re = psi[2*j], im = psi[2*j+1];
    real a = a0[j], b = MODEL == 0 ? (real)0 : b0[j];
#if MODEL != 3
    half_step(&a, &b, re*re+im*im, pump_value(p1, scale1, {p1_index}), dt*(real)0.5,
        R, gammaR, gammaI, gammaA, ria, rai, kappa);
#endif
    nonlinear(&re, &im, a, b, vr[{spatial_index}], vi[{spatial_index}], vi_scale, dt,
        inv_hbar, gC, gR, gI, R, gammaC);
#if MODEL != 3
    half_step(&a, &b, re*re+im*im, pump_value(p2, scale2, {p2_index}), dt*(real)0.5,
        R, gammaR, gammaI, gammaA, ria, rai, kappa);
    out_a[j] = a;
#if MODEL != 0
    out_b[j] = b;
#endif
#endif
    out_psi[2*j] = re;
    out_psi[2*j+1] = im;
}}
"""


def kernel_source(precision: str, model: str, batched: bool = False,
                  physics_fields=()) -> str:
    """Render one of the eight precision/model CUDA variants for RawKernel or NVRTC."""
    models = {"single": 0, "double": 1, "quadratic-double": 2, "nonlinear": 3}
    if precision not in {"single", "double"} or model not in models:
        raise ValueError(f"Invalid CUDA kernel variant: {precision}, {model}")
    source = _LOCAL_KERNEL.format(
        real="float" if precision == "single" else "double",
        model=models[model],
        mul="__fmul_rn" if precision == "single" else "__dmul_rn",
        spatial_index="j % spatial_n" if batched else "j",
        p1_index="j % p1_n" if batched else "j",
        p2_index="j % p2_n" if batched else "j",
    )
    if not physics_fields:
        return source
    names = {"hbar": "inv_hbar", "g_C": "gC", "g_R": "gR", "g_I": "gI",
             "R": "R", "gamma_C": "gammaC", "gamma_R": "gammaR",
             "gamma_I": "gammaI", "gamma_A": "gammaA", "R_IA": "ria",
             "R_AI": "rai", "kappa": "kappa"}
    source = source.replace(
        "real gammaR, real gammaI, real gammaA, real ria, real rai, real kappa) {\n    int j",
        "real gammaR, real gammaI, real gammaA, real ria, real rai, real kappa,\n"
        "    int potential_n, const real* batch_constants) {\n    int j",
    )
    source = source.replace("vr[j % spatial_n]", "vr[j % potential_n]")
    source = source.replace("vi[j % spatial_n]", "vi[j % potential_n]")
    head, body = source.split("    if (j >= n) return;", 1)
    for field in physics_fields:
        name = names[field]
        body = re.sub(rf"\b{name}\b", f"{name}_member", body)
    loads = "".join(
        f"\n    real {names[field]}_member = batch_constants[{index}*(n/spatial_n)+j/spatial_n];"
        for index, field in enumerate(physics_fields)
    )
    return head + "    if (j >= n) return;" + loads + body


@register_solver("split-step-fft-cuda")
class SplitStepFFTCudaSolver(AbstractSolver):
    """K-outer Strang split with two planned cuFFT transforms per step."""

    def __init__(self, config: Config, grid: SimulationGrid, batch_physics=None) -> None:
        super().__init__(config)
        self.batch_physics = BatchPhysics(config.physics, batch_physics, self.xp, getattr(config.solver, "precision", "double"), ndim=grid.ndim)
        if not compute_engine.use_gpu or getattr(self.xp, "__name__", "") != "cupy":
            raise RuntimeError("split-step-fft-cuda requires CuPy and an active CUDA GPU; use split-step-fft on CPU")
        if grid.boundary != "periodic":
            raise ValueError("split-step-fft-cuda requires a periodic grid")
        if self.batch_physics.any_nonzero("kinetic_relaxation_eta"):
            raise ValueError("split-step-fft-cuda does not support kinetic_relaxation_eta")

        import cupyx.scipy.fft as fft

        self._fft = fft
        self._ndim = grid.ndim
        self._fft_forward = fft.fft if grid.ndim == 1 else fft.fft2
        self._fft_inverse = fft.ifft if grid.ndim == 1 else fft.ifft2
        self._precision = config.solver.precision
        self._real_dtype = real_dtype(self.xp, self._precision)
        self._complex_dtype = complex_dtype(self.xp, self._precision)
        self._scalar = np.float32 if self._precision == "single" else np.float64
        kernel_fields = ("hbar", "g_C", "g_R", "g_I", "R", "gamma_C", "gamma_R",
                         "gamma_I", "gamma_A", "R_IA", "R_AI", "kappa")
        self._physics_fields = tuple(name for name in kernel_fields
                                     if name in self.batch_physics.host)
        if self._physics_fields:
            columns = []
            for name in self._physics_fields:
                values = self.batch_physics.host[name]
                columns.append(1.0 / values if name == "hbar" else values)
            self._physics_device = self.xp.asarray(np.stack(columns), dtype=self._real_dtype).ravel()
        self._shape = (1, 1, *grid.shape)
        self._batched = False
        self._spatial_n = int(np.prod(grid.shape))
        self._n = self._spatial_n
        self._plan_cache = {}
        self._kernel_cache = {}
        if "hbar" in self.batch_physics.host or "m_eff" in self.batch_physics.host:
            half = self.xp.exp(
                -1j * self.batch_physics.get("hbar") * grid.k_squared *
                (config.solver.dt / 2) / (2 * self.batch_physics.get("m_eff"))
            ).astype(self._complex_dtype)
        else:
            half = self.xp.exp(
                -1j * config.physics.hbar * grid.k_squared *
                (config.solver.dt / 2) / (2 * config.physics.m_eff)
            ).astype(self._complex_dtype)
        self._half = half
        self._half_normalized = (half / self._n).astype(self._complex_dtype)
        self._conj_half = self.xp.conj(half)
        k_squared = self.xp.asarray(grid.k_squared, dtype=self.xp.float64)
        if "hbar" in self.batch_physics.host or "m_eff" in self.batch_physics.host:
            self._full_normalized = (
                self.xp.exp(
                    -1j * self.batch_physics.get("hbar") * k_squared *
                    config.solver.dt / (2 * self.batch_physics.get("m_eff"))
                ) / self._n
            ).astype(self._complex_dtype)
        else:
            self._full_normalized = (
                self.xp.exp(
                    -1j * config.physics.hbar * k_squared *
                    config.solver.dt / (2 * config.physics.m_eff)
                ) / self._n
            ).astype(self._complex_dtype)
        self._potential_cached = None
        self._graph_cache = {}
        self._graph_signature = None
        self._graphs_failed = False

    def _plan(self, array):
        key = (array.shape, array.dtype.str, int(self.xp.cuda.runtime.getDevice()))
        if key not in self._plan_cache:
            # Plan from a C-contiguous template: callers may pass strided views.
            template = self.xp.empty(array.shape, dtype=array.dtype)
            self._plan_cache[key] = self._fft.get_fft_plan(template, axes=tuple(range(-self._ndim, 0)))
        return self._plan_cache[key]

    def _flat_gpu_input(self, array, dtype):
        if array.dtype != dtype:
            array = array.astype(dtype)
        return self.xp.ascontiguousarray(array).ravel()

    def _potential(self, potential):
        if potential is not self._potential_cached:
            re = self._flat_gpu_input(self.xp.real(potential), self._real_dtype)
            if self.xp.iscomplexobj(potential):
                im = self._flat_gpu_input(self.xp.imag(potential), self._real_dtype)
                scale = self._scalar(1)
            else:
                im, scale = re, self._scalar(0)
            self._potential_buffers = (re, im, scale)
            self._potential_cached = potential
        return self._potential_buffers

    def _kernel(self, model):
        key = (model, self._batched)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.xp.RawKernel(
                kernel_source(self._precision, model, key[1], self._physics_fields), "local_step",
                options=("--std=c++11",),
            )
        return self._kernel_cache[key]

    def _pump_pair(self, pump, t):
        times = (t + self.config.solver.dt / 4, t + 3 * self.config.solver.dt / 4)
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        term = pump.separable_term() if hasattr(pump, "separable_term") else None
        active_terms = terms if terms is not None else (term,) if term is not None else ()
        if self._batched and any(np.ndim(coefficient(t)) != 0
                                         for _, coefficient in active_terms):
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if getattr(self, "_pump_combiner", None) is None:
                self._pump_combiner = CudaPumpCombiner(self.xp, self._real_dtype, self._n)
            combiner = self._pump_combiner
            combiner.prepare(active_terms, 2, 2, shape=self._shape)
            stream = self.xp.cuda.get_current_stream()
            combiner.coefficients(active_terms, times, stream)
            stream.synchronize()
            return ((combiner.combine(0, 0), combiner.combine(1, 1)),
                    (self._scalar(1), self._scalar(1)))
        if term is None:
            return (tuple(self._flat_gpu_input(evaluate_pump(pump, time), self._real_dtype)
                          for time in times),
                    (self._scalar(1), self._scalar(1)))
        envelope, coefficient = term
        field = self._flat_gpu_input(envelope, self._real_dtype)
        return ((field, field), tuple(self._scalar(coefficient(time)) for time in times))

    def _launch(self, model, psi, fields, potential, pumps, scales):
        a = self._flat_gpu_input(fields[0], self._real_dtype)
        b = self._flat_gpu_input(fields[1], self._real_dtype) if len(fields) > 1 else a
        vr, vi, vi_scale = self._potential(potential)
        out = self.xp.empty_like(psi)
        out_a = self.xp.empty_like(a) if model != "nonlinear" else a
        out_b = self.xp.empty_like(b) if model not in {"single", "nonlinear"} else b
        p = self.config.physics
        scalar = self._scalar
        args = (psi.ravel().view(self._real_dtype), a, b, vr, vi, pumps[0], pumps[1],
                out.ravel().view(self._real_dtype), out_a, out_b, np.int32(self._n),
                np.int32(self._spatial_n), np.int32(pumps[0].size), np.int32(pumps[1].size),
                scales[0], scales[1], vi_scale, scalar(self.config.solver.dt),
                scalar(1 / p.hbar), scalar(p.g_C), scalar(p.g_R), scalar(p.g_I),
                scalar(p.R), scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
                scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI), scalar(p.kappa))
        if self._physics_fields:
            args += (np.int32(vr.size), self._physics_device)
        self._kernel(model)(((self._n + 255) // 256,), (256,), args)
        return out, tuple(x.reshape(self._shape) for x in (out_a, out_b)[:len(fields)])

    def _launch_into(self, model, psi, fields, potential, pumps, scales,
                     out, out_fields, graph_scales=None, scale_index=0):
        a = fields[0]
        b = fields[1] if len(fields) > 1 else a
        vr, vi, vi_scale = self._potential(potential)
        p = self.config.physics
        scalar = self._scalar
        if graph_scales is None:
            scale_args = (scales[0], scales[1])
            kernel = self._kernel(model)
        else:
            scale_args = (graph_scales, np.int32(scale_index))
            kernel = self._graph_kernel(model)
        args = (
            psi.ravel().view(self._real_dtype), a.ravel(), b.ravel(),
            vr, vi, pumps[0], pumps[1], out.ravel().view(self._real_dtype),
            out_fields[0].ravel(),
            out_fields[1].ravel() if len(fields) > 1 else out_fields[0].ravel(),
            np.int32(self._n), np.int32(self._spatial_n),
            np.int32(pumps[0].size), np.int32(pumps[1].size), *scale_args, vi_scale, scalar(self.config.solver.dt),
            scalar(1 / p.hbar), scalar(p.g_C), scalar(p.g_R), scalar(p.g_I),
            scalar(p.R), scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
            scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI), scalar(p.kappa),
        )
        if self._physics_fields:
            args += (np.int32(vr.size), self._physics_device)
        kernel(((self._n + 255) // 256,), (256,), args)

    def _graph_kernel(self, model):
        if not hasattr(self, "_graph_kernel_cache"):
            self._graph_kernel_cache = {}
        key = (model, self._batched)
        if key not in self._graph_kernel_cache:
            source = kernel_source(self._precision, model, key[1], self._physics_fields)
            source = source.replace(
                "real scale1, real scale2, real vi_scale,",
                "const real* scales, int scale_index, real vi_scale,",
            )
            source = source.replace("scale1", "scales[scale_index]")
            source = source.replace("scale2", "scales[scale_index+1]")
            kernel = self.xp.RawKernel(source, "local_step", options=("--std=c++11",))
            kernel.compile()
            self._graph_kernel_cache[key] = kernel
        return self._graph_kernel_cache[key]

    def _ensure_advance_buffers(self, n_fields):
        if (getattr(self, "_advance_buffers", None) is not None
                and self._advance_shape == self._shape
                and len(self._advance_buffers[1][0]) == n_fields):
            return
        xp = self.xp
        from cupy.cuda import cufft

        self._graph_cache.clear()
        self._graph_signature = None
        self._pump_combiner = None
        self._plan_cache.clear()
        self._advance_plan = None
        self._advance_stream = None
        self._graph_scales = None
        self._advance_buffers = None
        self._advance_shape = None
        spectrum = tuple(xp.empty(self._shape, self._complex_dtype) for _ in range(2))
        local = xp.empty(self._shape, self._complex_dtype)
        out = xp.empty_like(local)
        # The k-space product is consumed by the inverse FFT before the local
        # kernel overwrites out, so both can share one buffer.
        multiplied = out
        fields = tuple(tuple(xp.empty(self._shape, self._real_dtype)
                             for _ in range(n_fields)) for _ in range(2))
        graph_scales = xp.empty(64 * 2, self._real_dtype)
        advance_stream = xp.cuda.Stream(non_blocking=True)
        advance_plan = self._plan(spectrum[0])
        with advance_stream:
            xp.multiply(spectrum[0], self._full_normalized, out=multiplied)
            xp.multiply(spectrum[0], self._half, out=multiplied)
            advance_plan.fft(spectrum[0], multiplied, cufft.CUFFT_FORWARD)
            advance_plan.fft(multiplied, spectrum[0], cufft.CUFFT_INVERSE)
            advance_stream.synchronize()
        self._advance_buffers = (spectrum, fields, local, out, multiplied)
        self._graph_scales = graph_scales
        self._advance_stream = advance_stream
        self._advance_plan = advance_plan
        self._advance_shape = self._shape

    def _graph_for(self, length, model, potential, envelope, pump_flat,
                   combiner=None):
        signature = (self._shape, model, id(potential), id(envelope),
                     combiner.signature if combiner is not None else int(pump_flat.data.ptr))
        if signature != self._graph_signature:
            self._graph_cache.clear()
            self._graph_signature = signature
        if length in self._graph_cache:
            return self._graph_cache[length][0]
        xp = self.xp
        from cupy.cuda import cufft

        spectrum, fields, local, out, multiplied = self._advance_buffers
        scales = self._graph_scales
        plan = self._advance_plan
        self._potential(potential)
        self._graph_kernel(model)
        stream = self._advance_stream
        stream.synchronize()
        with stream:
            stream.begin_capture()
            try:
                for k in range(length):
                    xp.multiply(spectrum[k % 2], self._full_normalized, out=multiplied)
                    plan.fft(multiplied, local, cufft.CUFFT_INVERSE)
                    pumps = ((combiner.combine(2 * k, 0),
                              combiner.combine(2 * k + 1, 1))
                             if combiner is not None else (pump_flat, pump_flat))
                    self._launch_into(
                        model, local, fields[k % 2], potential,
                        pumps, None, out, fields[(k + 1) % 2],
                        scales, 2 * k,
                    )
                    plan.fft(out, spectrum[(k + 1) % 2], cufft.CUFFT_FORWARD)
                graph = stream.end_capture()
            except Exception:
                try:
                    stream.end_capture()
                except Exception:
                    pass
                raise
        if len(self._graph_cache) == 4:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[length] = (graph, combiner if combiner is not None else pump_flat)
        return graph

    def _activate_shape(self, state):
        shape = (state.field_k.shape if state.field_k is not None else state.field.shape)
        if shape != self._shape or state.batched != self._batched:
            self._shape = shape
            self._batched = state.batched
            self._n = int(np.prod(shape))
            self._graph_cache.clear()
            self._graph_signature = None
            self._pump_combiner = None
            if hasattr(self, "_zero_pump"):
                del self._zero_pump

    def advance(self, potential, pump, reservoir, boundary_condition, state,
                start_step: int, n_steps: int) -> None:
        """Run supported periodic chunks with planned FFTs and optional graphs."""
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        if n_steps <= 0:
            return
        self._activate_shape(state)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir

        models = {SingleReservoir: "single", DoubleReservoir: "double",
                  QuadraticDoubleReservoir: "quadratic-double"}
        model = models.get(type(reservoir))
        diffusion = any(reservoir.batch_physics.any_nonzero(name) for name in
                        ("reservoir_diffusion_R", "reservoir_diffusion_A",
                         "reservoir_diffusion_I")) if model is not None else False
        absorption = getattr(boundary_condition, "absorption", None)
        if model is None or diffusion or not getattr(absorption, "after_step_is_noop", False):
            return super().advance(potential, pump, reservoir, boundary_condition,
                                   state, start_step, n_steps)

        xp = self.xp
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        term = (terms[0] if terms is not None and len(terms) == 1 else
                pump.separable_term() if terms is None and hasattr(pump, "separable_term") else None)
        if terms is None and term is not None:
            terms = (term,)
        multi = terms is not None and (len(terms) > 1 or
                (self._batched and len(terms) == 1 and
                 np.ndim(terms[0][1](state.t)) != 0))
        no_lasers = not getattr(pump, "lasers", [True])
        if term is not None:
            envelope, coefficient = term
            pump_flat = self._flat_gpu_input(envelope, self._real_dtype)
        elif no_lasers:
            if not hasattr(self, "_zero_pump"):
                self._zero_pump = xp.zeros(self._shape, dtype=self._real_dtype)
            envelope = self._zero_pump
            pump_flat = self._flat_gpu_input(envelope, self._real_dtype)
            coefficient = lambda t: 1
        else:
            envelope = None
            coefficient = None
            pump_flat = None

        spectrum_in = getattr(state, "field_k", None)
        if spectrum_in is None:
            spectrum_in = self._fft_forward(
                xp.ascontiguousarray(state.field), plan=self._plan(state.field)
            )
        spectrum_in = spectrum_in.astype(self._complex_dtype, copy=False)
        fields_in = reservoir.get_state()
        xp.cuda.get_current_stream().synchronize()
        self._ensure_advance_buffers(len(fields_in))
        if multi:
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if getattr(self, "_pump_combiner", None) is None:
                self._pump_combiner = CudaPumpCombiner(xp, self._real_dtype, self._n)
            combiner = self._pump_combiner
            combiner.prepare(terms, 2, 2 * 64, shape=self._shape if self._batched else None)
            xp.cuda.get_current_stream().synchronize()
        else:
            combiner = None
        spec, res, local, out, multiplied = self._advance_buffers
        stream = self._advance_stream
        plan = self._advance_plan
        done = 0
        with stream:
            xp.multiply(spectrum_in, self._conj_half, out=spec[0])
            for target, source in zip(res[0], fields_in):
                xp.copyto(target, source)
            pending_hosts = []
            if (self.config.solver.cuda_graphs and not self._graphs_failed
                    and (coefficient is not None or multi)):
                try:
                    while n_steps - done >= 2:
                        remaining = n_steps - done
                        length = min(64, remaining - remaining % 2)
                        graph = self._graph_for(
                            length, model, potential, envelope, pump_flat, combiner
                        )
                        times = [time for k in range(length) for time in (
                            state.t0 + (start_step + done + k) * self.config.solver.dt
                            + self.config.solver.dt / 4,
                            state.t0 + (start_step + done + k) * self.config.solver.dt
                            + 3 * self.config.solver.dt / 4)]
                        if multi:
                            pending_hosts.append(combiner.coefficients(terms, times, stream))
                            host_scales = self._scalar([1] * (length * 2))
                        else:
                            host_scales = self._scalar([coefficient(t) for t in times])
                        pending_hosts.append(host_scales)
                        self._graph_scales[:length * 2].set(host_scales, stream=stream)
                        graph.launch(stream=stream)
                        done += length
                        if len(pending_hosts) >= 32:
                            stream.synchronize()
                            pending_hosts.clear()
                except Exception as exc:
                    import sys
                    self._graphs_failed = True
                    self._graph_cache.clear()
                    self._graph_signature = None
                    print(f"split-step-fft-cuda: CUDA graphs unavailable ({exc}); using regular launches.",
                          file=sys.stderr)
                    stream.synchronize()
                    xp.multiply(spectrum_in, self._conj_half, out=spec[0])
                    for target, source in zip(res[0], fields_in):
                        xp.copyto(target, source)
                    done = 0

            if done < n_steps:
                from cupy.cuda import cufft
                for k in range(done, n_steps):
                    parity = (k - done) % 2
                    xp.multiply(spec[parity], self._full_normalized, out=multiplied)
                    plan.fft(multiplied, local, cufft.CUFFT_INVERSE)
                    t = state.t0 + (start_step + k) * self.config.solver.dt
                    times = (t + self.config.solver.dt / 4,
                             t + 3 * self.config.solver.dt / 4)
                    if multi:
                        pending_hosts.append(combiner.coefficients(terms, times, stream))
                        pumps = (combiner.combine(0, 0), combiner.combine(1, 1))
                        factors = (self._scalar(1), self._scalar(1))
                    elif coefficient is None:
                        pumps = tuple(self._flat_gpu_input(evaluate_pump(pump, time), self._real_dtype)
                                      for time in times)
                        factors = (self._scalar(1), self._scalar(1))
                    else:
                        pumps = (pump_flat, pump_flat)
                        factors = tuple(self._scalar(coefficient(time)) for time in times)
                    self._launch_into(model, local, res[parity], potential, pumps,
                                      factors, out, res[1 - parity])
                    plan.fft(out, spec[1 - parity], cufft.CUFFT_FORWARD)
                    if len(pending_hosts) >= 32:
                        stream.synchronize()
                        pending_hosts.clear()

            final_index = (n_steps - done) % 2
            xp.multiply(spec[final_index], self._half, out=spec[final_index])
            state.field = None
            state.field_k = spec[final_index].copy()
            reservoir.set_state(tuple(field.copy() for field in res[final_index]))
            stream.synchronize()
        state.t = state.t0 + (start_step + n_steps) * self.config.solver.dt

    def step(self, potential, pump, reservoir: AbstractReservoir,
             boundary_condition: BoundaryCondition, state: SimulationState) -> None:
        """Advance K(dt/2) [R(dt/2) N(dt) R(dt/2)] K(dt/2)."""
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        self._activate_shape(state)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir

        psi_k = getattr(state, "field_k", None)
        if psi_k is None:
            psi_k = self._fft_forward(self.xp.ascontiguousarray(state.field), plan=self._plan(state.field))
        psi_k = psi_k.astype(self._complex_dtype, copy=False)
        psi = self._fft_inverse(psi_k * self._half_normalized, norm="forward",
                              plan=self._plan(psi_k))
        reservoir_state = reservoir.get_state()
        cls = type(reservoir)
        model = ({SingleReservoir: "single", DoubleReservoir: "double",
                  QuadraticDoubleReservoir: "quadratic-double"}).get(cls)
        diffusion = any(reservoir.batch_physics.any_nonzero(name) for name in
                        ("reservoir_diffusion_R", "reservoir_diffusion_A",
                         "reservoir_diffusion_I")) if model is not None else False
        if model is not None and not diffusion:
            pumps, scales = self._pump_pair(pump, state.t)
            psi, fields = self._launch(model, psi, reservoir_state, potential, pumps, scales)
            reservoir.set_state(fields)
        else:
            # Built-in diffusion uses periodic real_laplacian in reservoir.step;
            # plugin reservoirs keep their own RK2 or other step semantics.
            dt = self.config.solver.dt
            p1, p2 = (evaluate_pump(pump, state.t + dt / 4),
                      evaluate_pump(pump, state.t + 3 * dt / 4))
            reservoir.step(dt / 2, psi, p1)
            dummy = self._flat_gpu_input(p1, self._real_dtype)
            current = reservoir.get_state()
            densities = (reservoir.get_active_density(current),
                         reservoir.get_inactive_density(current))
            psi, _ = self._launch("nonlinear", psi, densities, potential,
                                  (dummy, dummy), (self._scalar(1), self._scalar(1)))
            reservoir.step(dt / 2, psi, p2)
        psi_k = self._fft_forward(psi, plan=self._plan(psi)) * self._half
        absorption = getattr(boundary_condition, "absorption", None)
        if getattr(absorption, "after_step_is_noop", False):
            state.field = None
            state.field_k = psi_k
        else:
            materialized = self._fft_inverse(psi_k, plan=self._plan(psi_k))
            state.field = boundary_condition.after_step_action(materialized)
            psi = self.xp.ascontiguousarray(state.field)
            state.field_k = self._fft_forward(psi, plan=self._plan(psi))
