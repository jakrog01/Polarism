"""Two-transform CUDA Strang solver with local reservoir and condensate steps."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid_2d import SimulationGrid2D
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
    int n, real scale1, real scale2, real vi_scale, real dt,
    real inv_hbar, real gC, real gR, real gI, real R, real gammaC,
    real gammaR, real gammaI, real gammaA, real ria, real rai, real kappa) {{
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n) return;
    real re = psi[2*j], im = psi[2*j+1];
    real a = a0[j], b = MODEL == 0 ? (real)0 : b0[j];
#if MODEL != 3
    half_step(&a, &b, re*re+im*im, pump_value(p1, scale1, j), dt*(real)0.5,
        R, gammaR, gammaI, gammaA, ria, rai, kappa);
#endif
    nonlinear(&re, &im, a, b, vr[j], vi[j], vi_scale, dt,
        inv_hbar, gC, gR, gI, R, gammaC);
#if MODEL != 3
    half_step(&a, &b, re*re+im*im, pump_value(p2, scale2, j), dt*(real)0.5,
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


def kernel_source(precision: str, model: str) -> str:
    """Render one of the eight precision/model CUDA variants for RawKernel or NVRTC."""
    models = {"single": 0, "double": 1, "quadratic-double": 2, "nonlinear": 3}
    if precision not in {"single", "double"} or model not in models:
        raise ValueError(f"Invalid CUDA kernel variant: {precision}, {model}")
    return _LOCAL_KERNEL.format(
        real="float" if precision == "single" else "double",
        model=models[model],
        mul="__fmul_rn" if precision == "single" else "__dmul_rn",
    )


@register_solver("split-step-fft-cuda")
class SplitStepFFTCudaSolver(AbstractSolver):
    """K-outer Strang split with two planned cuFFT transforms per step."""

    def __init__(self, config: Config, grid: SimulationGrid2D) -> None:
        super().__init__(config)
        if not compute_engine.use_gpu or getattr(self.xp, "__name__", "") != "cupy":
            raise RuntimeError("split-step-fft-cuda requires CuPy and an active CUDA GPU; use split-step-fft on CPU")
        if grid.boundary != "periodic":
            raise ValueError("split-step-fft-cuda requires a periodic grid")
        if getattr(config.physics, "kinetic_relaxation_eta", 0.0) != 0:
            raise ValueError("split-step-fft-cuda does not support kinetic_relaxation_eta")

        import cupyx.scipy.fft as fft

        self._fft = fft
        self._precision = config.solver.precision
        self._real_dtype = real_dtype(self.xp, self._precision)
        self._complex_dtype = complex_dtype(self.xp, self._precision)
        self._scalar = np.float32 if self._precision == "single" else np.float64
        self._shape = (grid.ny, grid.nx)
        self._n = grid.nx * grid.ny
        self._plan_cache = {}
        self._kernel_cache = {}
        half = self.xp.exp(
            -1j * config.physics.hbar * grid.k_squared *
            (config.solver.dt / 2) / (2 * config.physics.m_eff)
        ).astype(self._complex_dtype)
        self._half = half
        self._half_normalized = (half / self._n).astype(self._complex_dtype)
        self._potential_cached = None

    def _plan(self, array):
        key = (array.shape, array.dtype.str, int(self.xp.cuda.runtime.getDevice()))
        if key not in self._plan_cache:
            self._plan_cache[key] = self._fft.get_fft_plan(array, axes=(-2, -1))
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
        if model not in self._kernel_cache:
            self._kernel_cache[model] = self.xp.RawKernel(
                kernel_source(self._precision, model), "local_step",
                options=("--std=c++11",),
            )
        return self._kernel_cache[model]

    def _pump_pair(self, pump, t):
        times = (t + self.config.solver.dt / 4, t + 3 * self.config.solver.dt / 4)
        term = pump.separable_term() if hasattr(pump, "separable_term") else None
        if term is None:
            return (tuple(self._flat_gpu_input(pump(time), self._real_dtype) for time in times),
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
                scales[0], scales[1], vi_scale, scalar(self.config.solver.dt),
                scalar(1 / p.hbar), scalar(p.g_C), scalar(p.g_R), scalar(p.g_I),
                scalar(p.R), scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
                scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI), scalar(p.kappa))
        self._kernel(model)(((self._n + 255) // 256,), (256,), args)
        return out, tuple(x.reshape(self._shape) for x in (out_a, out_b)[:len(fields)])

    def step(self, potential, pump, reservoir: AbstractReservoir,
             boundary_condition: BoundaryCondition, state: SimulationState) -> None:
        """Advance K(dt/2) [R(dt/2) N(dt) R(dt/2)] K(dt/2)."""
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir

        psi_k = getattr(state, "psi_k", None)
        if psi_k is None:
            psi_k = self._fft.fft2(self.xp.ascontiguousarray(state.psi), plan=self._plan(state.psi))
        psi_k = psi_k.astype(self._complex_dtype, copy=False)
        psi = self._fft.ifft2(psi_k * self._half_normalized, norm="forward",
                              plan=self._plan(psi_k))
        reservoir_state = reservoir.get_state()
        cls = type(reservoir)
        model = ({SingleReservoir: "single", DoubleReservoir: "double",
                  QuadraticDoubleReservoir: "quadratic-double"}).get(cls)
        diffusion = any(getattr(reservoir, name, 0.0) != 0.0 for name in
                        ("reservoir_diffusion_R", "reservoir_diffusion_A",
                         "reservoir_diffusion_I"))
        if model is not None and not diffusion:
            pumps, scales = self._pump_pair(pump, state.t)
            psi, fields = self._launch(model, psi, reservoir_state, potential, pumps, scales)
            reservoir.set_state(fields)
        else:
            # Built-in diffusion uses periodic real_laplacian in reservoir.step;
            # plugin reservoirs keep their own RK2 or other step semantics.
            dt = self.config.solver.dt
            p1, p2 = (pump(state.t + dt / 4), pump(state.t + 3 * dt / 4))
            reservoir.step(dt / 2, psi, p1)
            dummy = self._flat_gpu_input(p1, self._real_dtype)
            current = reservoir.get_state()
            densities = (reservoir.get_active_density(current),
                         reservoir.get_inactive_density(current))
            psi, _ = self._launch("nonlinear", psi, densities, potential,
                                  (dummy, dummy), (self._scalar(1), self._scalar(1)))
            reservoir.step(dt / 2, psi, p2)
        psi_k = self._fft.fft2(psi, plan=self._plan(psi)) * self._half
        absorption = getattr(boundary_condition, "absorption", None)
        if getattr(absorption, "after_step_is_noop", False):
            state.psi = None
            state.psi_k = psi_k
        else:
            materialized = self._fft.ifft2(psi_k, plan=self._plan(psi_k))
            state.psi = boundary_condition.after_step_action(materialized)
            state.psi_k = self._fft.fft2(state.psi, plan=self._plan(state.psi))
