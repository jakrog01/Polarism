"""Two-transform CUDA Strang solver with local reservoir and condensate steps."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import re

from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver, ACCEPTED, SolverCapabilities
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid import SimulationGrid
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


def spinor_matrix_kernel_source(precision: str) -> str:
    """Render a batched two-spectrum complex matrix product."""
    if precision not in ("single", "double"):
        raise ValueError("Invalid spinor spectral precision")
    real = "float" if precision == "single" else "double"
    return f'''typedef {real} real;
extern "C" __global__ void spinor_matrix_product(
    const real* matrix, const real* field, real* output,
    int spatial_n, int total_n, int matrix_members) {{
    int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j >= total_n) return;
    int member = j/spatial_n, cell = j%spatial_n;
    int a = 2*(member*2*spatial_n+cell);
    int b = a+2*spatial_n;
    int m = 2*((matrix_members == 1 ? 0 : member)*4*spatial_n+cell);
    real ar = field[a], ai = field[a+1], br = field[b], bi = field[b+1];
    real x0 = matrix[m], y0 = matrix[m+1];
    real x1 = matrix[m+2*spatial_n], y1 = matrix[m+2*spatial_n+1];
    real x2 = matrix[m+4*spatial_n], y2 = matrix[m+4*spatial_n+1];
    real x3 = matrix[m+6*spatial_n], y3 = matrix[m+6*spatial_n+1];
    output[a] = x0*ar-y0*ai+x1*br-y1*bi;
    output[a+1] = x0*ai+y0*ar+x1*bi+y1*br;
    output[b] = x2*ar-y2*ai+x3*br-y3*bi;
    output[b+1] = x2*ai+y2*ar+x3*bi+y3*br;
}}
'''


def apply_spinor_matrix_cuda(kernel, matrix, field, xp):
    """Apply a contiguous 2×2 Fourier-space matrix to two components."""
    field = field.astype(matrix.dtype, copy=False)
    spatial_n = int(np.prod(field.shape[2:]))
    total_n = field.shape[0] * spatial_n
    output = xp.empty_like(field)
    real_dtype = field.real.dtype
    kernel(((total_n + 255) // 256,), (256,),
           (xp.ascontiguousarray(matrix).view(real_dtype).ravel(),
            xp.ascontiguousarray(field).view(real_dtype).ravel(),
            output.view(real_dtype).ravel(), np.int32(spatial_n), np.int32(total_n),
            np.int32(matrix.shape[0])))
    return output


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
    """Render a precision and reservoir CUDA variant for RawKernel or NVRTC."""
    models = {"single": 0, "double": 1, "quadratic-double": 2,
              "nonlinear": 3, "none": 3}
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
    if model == "none":
        source = source.replace("    real a = a0[j], b = MODEL == 0 ? (real)0 : b0[j];",
                                "    real a = (real)0, b = (real)0;")
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


def _feature_source(source, real, batched, terms, varying_potential, drive):
    declarations = []
    if varying_potential:
        declarations.extend(("const real* cap_re", "const real* cap_im"))
    if drive:
        declarations.extend(("const real* drive_re", "const real* drive_im",
                             "const real* drive_energy"))
    if terms:
        declarations.extend(("const real* x_coords", "const real* y_coords"))
    loads = []
    for number, term in enumerate(terms):
        for ctype, name, *layout in term.cuda.arguments:
            kind = layout[0] if layout else "raw"
            name_in_kernel = f"term_{number}_{name}"
            type_in_kernel = ctype.replace("{REAL}", real)
            if kind == "raw":
                declarations.append(f"{type_in_kernel} {name_in_kernel}")
            else:
                declarations.append(f"{type_in_kernel} {name_in_kernel}_arg")
                value_type = type_in_kernel.replace("const", "").replace("*", "").strip()
                index = "cell" if kind == "shared" else "j"
                loads.append(f"    {value_type} {name_in_kernel} = {name_in_kernel}_arg[{index}];\n")
    declarations.extend(("const real* feature_values", "int feature_index",
                         "int feature_members", "int feature_vr_n"))
    start = source.index('extern "C" __global__ void local_step(')
    end = source.index(") {", start)
    source = source[:end] + ",\n    " + ", ".join(declarations) + source[end:]
    local = ("    int cell = j % spatial_n;\n"
             "    int member = j / spatial_n;\n"
             "    int feature_offset = (feature_index*feature_members+member)*3;\n"
             "    real potential_scale = feature_values[feature_offset];\n"
             "    real drive_scale = feature_values[feature_offset+1];\n"
             "    real t_stage = feature_values[feature_offset+2];\n")
    if terms:
        local += ("    real x = x_coords[cell], y = y_coords[cell];\n"
                  "    real n_active = a, n_inactive = b;\n")
        local += "".join(loads)
    potential_index = "j % feature_vr_n"
    local += (f"    real local_vr = vr[{potential_index}], "
              f"local_vi = vi[{potential_index}];\n")
    if varying_potential:
        local += ("    local_vr = local_vr*potential_scale + cap_re[j];\n"
                  "    local_vi = local_vi*potential_scale + cap_im[j];\n")
    if drive:
        local += "    local_vr -= drive_energy[j];\n"
        local += ("    re += (real)0.5*dt*inv_hbar*drive_scale*drive_im[j];\n"
                  "    im -= (real)0.5*dt*inv_hbar*drive_scale*drive_re[j];\n")
    anchor = "    nonlinear(&re, &im, a, b, "
    start = source.index(anchor)
    end = source.index(";", start) + 1
    source = source[:start] + local + ("    nonlinear(&re, &im, a, b, local_vr, local_vi, "
              "vi_scale, dt,\n        inv_hbar, gC, gR, gI, R, gammaC);") + source[end:]
    insert = ""
    if terms:
        insert += "    real pre = re, pim = im, rho = pre*pre+pim*pim;\n"
        for category in ("energy", "rate"):
            expressions = []
            for number, term in enumerate(terms):
                if term.category != category:
                    continue
                expression = term.cuda.expression
                for argument in term.cuda.arguments:
                    expression = re.sub(rf"\b{argument[1]}\b",
                                        f"term_{number}_{argument[1]}", expression)
                expressions.append(f"({expression})")
            if expressions:
                insert += f"    real extra_{category} = " + " + ".join(expressions) + ";\n"
        energy = "extra_energy" if any(term.category == "energy" for term in terms) else "(real)0"
        rate = "extra_rate" if any(term.category == "rate" for term in terms) else "(real)0"
        insert += (f"    real extra_amp = exp({rate}*dt);\n"
                   f"    real extra_phase = {energy}*inv_hbar*dt;\n"
                   "    real extra_s, extra_c;\n"
                   "    sincos(extra_phase, &extra_s, &extra_c);\n"
                   "    re = extra_amp*(pre*extra_c+pim*extra_s);\n"
                   "    im = extra_amp*(pim*extra_c-pre*extra_s);\n")
    if drive:
        insert += ("    re += (real)0.5*dt*inv_hbar*drive_scale*drive_im[j];\n"
                   "    im -= (real)0.5*dt*inv_hbar*drive_scale*drive_re[j];\n")
    location = source.index("\n#if MODEL != 3\n    half_step", source.index("    nonlinear(&re"))
    return source[:location] + "\n" + insert.rstrip() + source[location:]


@register_solver("split-step-fft-cuda")
class SplitStepFFTCudaSolver(AbstractSolver):
    """K-outer Strang split with two planned cuFFT transforms per step.

    The Wigner increment is added inside the local kernel, after the second
    reservoir half-step and before the second kinetic half-step.
    """
    injects_noise = True
    capabilities = SolverCapabilities(
        components=frozenset({1, 2}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="quantitative", potential=True,
        kinetic_relaxation=False, dispersion=frozenset({"parabolic"}), terms=True,
        coherent_drive=True, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='GPU-only, two-transform spectral Strang splitting with symmetric reservoir half-steps. Periodic grids only; no kinetic relaxation.',
    )

    def __init__(self, config: Config, grid: SimulationGrid, model) -> None:
        super().__init__(config, grid, model)
        if not compute_engine.use_gpu or getattr(self.xp, "__name__", "") != "cupy":
            raise RuntimeError("split-step-fft-cuda requires CuPy and an active CUDA GPU; use split-step-fft on CPU")
        if grid.boundary != "periodic":
            raise ValueError("split-step-fft-cuda requires a periodic grid")
        if self.batch_physics.any_nonzero("kinetic_relaxation_eta"):
            raise ValueError("split-step-fft-cuda does not support kinetic_relaxation_eta")

        self._spinor = model.components == 2
        if self._spinor:
            from polarism.solver.split_step_fft_solver import SplitStepFFTSolver

            self._spinor_delegate = SplitStepFFTSolver(config, grid, model)
            self._spinor_matrix_kernel = self.xp.RawKernel(
                spinor_matrix_kernel_source(config.solver.precision), "spinor_matrix_product")
            self._spinor_delegate._kinetic_half_step_fft = self._spinor_half_step_fft
            return

        import cupyx.scipy.fft as fft

        self._fft = fft
        self._ndim = grid.ndim
        self._fft_forward = fft.fft if grid.ndim == 1 else fft.fft2
        self._fft_inverse = fft.ifft if grid.ndim == 1 else fft.ifft2
        self._precision = config.solver.precision
        self._real_dtype = real_dtype(self.xp, self._precision)
        self._complex_dtype = complex_dtype(self.xp, self._precision)
        self._scalar = np.float32 if self._precision == "single" else np.float64
        self._model_features = model
        self._has_features = bool(model.terms or model.drive is not None or
                                  model.time_dependent_potential)
        for term in model.terms:
            if term.cuda is None:
                raise ValueError(f"Term {type(term).__name__} has no CUDA snippet")
        self._feature_static = ()
        self._feature_buffer = None
        self._noise = model.noise if model.noise is not None and model.noise.active else None
        self._noise_step_base = None
        self._launch_step = 0
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
        self._zero_res = self.xp.zeros(self._shape, self._real_dtype)
        if self._has_features:
            self._prepare_features()
        if self._noise is not None:
            self._noise_step_base = self.xp.zeros(1, dtype=self.xp.uint64)

    def _spinor_half_step_fft(self, field):
        delegate = self._spinor_delegate
        spectrum = delegate._fft(field)
        evolved = apply_spinor_matrix_cuda(self._spinor_matrix_kernel,
                                            delegate._kinetic_propagator_half,
                                            spectrum, self.xp)
        return delegate._ifft(evolved)

    def _step_spinor(self, model, state, dt):
        xp = self.xp
        delegate = self._spinor_delegate
        reservoir = model.reservoir
        t = state.t
        dt = self.config.solver.dt
        midpoint = t + dt / 2
        hbar = self.batch_physics.get("hbar")
        delegate._reservoir_half_step(model, dt / 2, state.field,
                                      evaluate_pump(model.pump, t + dt / 4))
        res_state = reservoir.get_state()
        active = reservoir.get_active_density(res_state)
        inactive = reservoir.get_inactive_density(res_state)
        field = self._spinor_half_step_fft(state.field)
        source_half = (0 if model.drive is None else
                       (-0.5j * dt / hbar) * model.drive.spatial * model.drive.envelope(midpoint))
        field = (field + source_half).astype(delegate._kinetic_propagator_half.dtype,
                                            copy=False)
        potential = model.potential_at(midpoint) - (0 if model.drive is None else model.drive.energy)
        rho = xp.abs(field) ** 2
        gain = (self.batch_physics.get("R") * active - self.batch_physics.get("gamma_C")) / 2
        gain = gain + xp.imag(potential) / hbar
        x = 2 * gain * dt
        small = xp.abs(x) < 1e-6
        phase_time = dt * xp.where(small, 1 + x / 2 + x * x / 6,
                                   xp.expm1(x) / xp.where(small, 1, x))
        phase = ((xp.real(potential) + self.batch_physics.get("g_R") * active
                  + self.batch_physics.get("g_I") * inactive) * dt
                 + self.batch_physics.get("g_C") * rho * phase_time) / hbar
        other_phase_time = xp.broadcast_to(phase_time, rho.shape)[:, ::-1]
        phase = phase + self.batch_physics.get("alpha_2") * rho[:, ::-1] * other_phase_time / hbar
        field = field * xp.exp(gain * dt - 1j * phase)
        if model.terms:
            energy, rate = model.extra_coefficients(field, res_state, midpoint)
            field = field * xp.exp((rate - 1j * energy / hbar) * dt)
        field = (field + source_half).astype(delegate._kinetic_propagator_half.dtype,
                                            copy=False)
        if model.noise is not None:
            model.noise.apply_arrays(field, active, state.step)
        field = self._spinor_half_step_fft(field)
        delegate._reservoir_half_step(model, dt / 2, field,
                                      evaluate_pump(model.pump, t + 3 * dt / 4))
        state.field = model.boundary.after_step_action(field)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED

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
                im, scale = (self.xp.zeros_like(re), self._scalar(1)) if self._model_features.time_dependent_potential else (re, self._scalar(0))
            self._potential_buffers = (re, im, scale)
            self._potential_cached = potential
        return self._potential_buffers

    def _prepare_features(self):
        model = self._model_features
        xp = self.xp
        shape = self._shape
        spatial = self._spatial_n
        arrays = []
        if model.time_dependent_potential:
            cap = xp.broadcast_to(xp.asarray(model.cap), shape)
            arrays.extend((self._flat_gpu_input(xp.real(cap), self._real_dtype),
                           self._flat_gpu_input(xp.imag(cap), self._real_dtype)))
        if model.drive is not None:
            drive = xp.broadcast_to(xp.asarray(model.drive.spatial), shape)
            energy = xp.broadcast_to(xp.asarray(model.drive.energy), shape)
            arrays.extend((self._flat_gpu_input(xp.real(drive), self._real_dtype),
                           self._flat_gpu_input(xp.imag(drive), self._real_dtype),
                           self._flat_gpu_input(energy, self._real_dtype)))
        if model.terms:
            arrays.extend((self._flat_gpu_input(xp.asarray(model.grid.X), self._real_dtype),
                           self._flat_gpu_input(xp.asarray(model.grid.Y), self._real_dtype)
                           if model.grid.ndim == 2 else xp.zeros(spatial, self._real_dtype)))
        for term in model.terms:
            values = term.cuda_values()
            for ctype, name, *layout in term.cuda.arguments:
                value = values[name]
                if "*" in ctype:
                    dtype = (xp.uint64 if "long long" in ctype else
                             xp.uint32 if "unsigned int" in ctype else
                             xp.int32 if "int" in ctype else
                             xp.float64 if "double" in ctype else
                             xp.float32 if "float" in ctype else self._real_dtype)
                    arrays.append(xp.ascontiguousarray(value, dtype=dtype).ravel())
                elif "int" in ctype:
                    arrays.append(np.int32(value))
                else:
                    arrays.append(np.float64(value) if "double" in ctype else
                                  np.float32(value) if "float" in ctype else self._scalar(value))
        self._feature_static = tuple(arrays)
        self._feature_buffer = xp.empty(64 * (self._n // spatial) * 3, self._real_dtype)

    def _feature_values(self, times, stream=None):
        model = self._model_features
        members = self._n // self._spatial_n
        values = np.empty((len(times), members, 3), dtype=self._scalar)
        def sample(value):
            host = np.asarray(value.get() if hasattr(value, "get") else value)
            return np.broadcast_to(host.reshape(-1), (members,))
        for index, time in enumerate(times):
            values[index, :, 0] = (sample(model.potential_envelope(time))
                                   if model.time_dependent_potential else 1)
            values[index, :, 1] = (sample(model.drive.envelope(time))
                                   if model.drive is not None else 0)
            values[index, :, 2] = time
        self._feature_buffer[:values.size].set(values.ravel(), stream=stream)
        return values

    def _source(self, model, graph=False):
        source = kernel_source(self._precision, model, self._batched, self._physics_fields)
        if graph:
            source = source.replace("real scale1, real scale2, real vi_scale,",
                                    "const real* scales, int scale_index, real vi_scale,")
            source = source.replace("scale1", "scales[scale_index]")
            source = source.replace("scale2", "scales[scale_index+1]")
        if self._has_features:
            source = _feature_source(source, "float" if self._precision == "single" else "double",
                                     self._batched, self._model_features.terms,
                                     self._model_features.time_dependent_potential,
                                     self._model_features.drive is not None)
        if self._noise_in_kernel(model):
            source = self._noise_source(source, graph)
        return source

    def _noise_in_kernel(self, model):
        return self._noise is not None and model != "nonlinear"

    def _noise_source(self, source, graph):
        from polarism.noise.philox import device_source
        real = "float" if self._precision == "single" else "double"
        step = ("const unsigned long long* noise_step_base, unsigned int noise_step_offset"
                if graph else "unsigned long long noise_step")
        declarations = (f"const unsigned long long* noise_seeds, const {real}* noise_gamma, "
                        f"const {real}* noise_scattering, const {real}* noise_strength, "
                        f"{step}, {real} noise_scale")
        start = source.index('extern "C" __global__ void local_step(')
        end = source.index(") {", start)
        source = source[:end] + ",\n    " + declarations + source[end:]
        source = source[:start] + device_source(self._precision) + source[start:]
        step_value = "noise_step_base[0] + noise_step_offset" if graph else "noise_step"
        injection = ("    {\n"
                     "        int noise_member = j / spatial_n;\n"
                     "        wigner_add(&re, &im, noise_seeds[noise_member], (unsigned int)(j % spatial_n),\n"
                     f"            {step_value}, noise_strength[noise_member], noise_gamma[noise_member],\n"
                     "            noise_scattering[noise_member], a, noise_scale);\n"
                     "    }\n")
        anchor = "    out_psi[2*j] = re;"
        return source.replace(anchor, injection + anchor)

    def _noise_arguments(self, model, step, graph):
        if not self._noise_in_kernel(model):
            return ()
        noise = self._noise
        return (noise._device_seeds, noise._device_gamma, noise._device_scattering,
                noise._device_strength,
                *((self._noise_step_base, np.uint32(step)) if graph else (np.uint64(step),)),
                self._scalar(noise.dt / (4.0 * noise.volume)))

    def _kernel(self, model):
        key = (model, self._batched)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.xp.RawKernel(
                self._source(model), "local_step",
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
        a = self._flat_gpu_input(fields[0], self._real_dtype) if fields else self._zero_res.ravel()
        b = self._flat_gpu_input(fields[1], self._real_dtype) if len(fields) > 1 else a
        vr, vi, vi_scale = self._potential(potential)
        if self._model_features.time_dependent_potential:
            vi_scale = self._scalar(1)
        out = self.xp.empty_like(psi)
        out_a = self.xp.empty_like(a) if model not in {"nonlinear", "none"} else a
        out_b = self.xp.empty_like(b) if model not in {"single", "nonlinear", "none"} else b
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
        if self._has_features:
            self._feature_values((self._launch_time + self.config.solver.dt / 2,))
            args += (*self._feature_static, self._feature_buffer, np.int32(0),
                     np.int32(self._n // self._spatial_n), np.int32(vr.size))
        args += self._noise_arguments(model, self._launch_step, False)
        self._kernel(model)(((self._n + 255) // 256,), (256,), args)
        return out, tuple(x.reshape(self._shape) for x in (out_a, out_b)[:len(fields)])

    def _launch_into(self, model, psi, fields, potential, pumps, scales,
                     out, out_fields, graph_scales=None, scale_index=0, noise_step=0):
        a = fields[0] if fields else self._zero_res
        b = fields[1] if len(fields) > 1 else a
        vr, vi, vi_scale = self._potential(potential)
        if self._model_features.time_dependent_potential:
            vi_scale = self._scalar(1)
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
            out_fields[0].ravel() if fields else self._zero_res.ravel(),
            out_fields[1].ravel() if len(fields) > 1 else
            out_fields[0].ravel() if fields else self._zero_res.ravel(),
            np.int32(self._n), np.int32(self._spatial_n),
            np.int32(pumps[0].size), np.int32(pumps[1].size), *scale_args, vi_scale, scalar(self.config.solver.dt),
            scalar(1 / p.hbar), scalar(p.g_C), scalar(p.g_R), scalar(p.g_I),
            scalar(p.R), scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
            scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI), scalar(p.kappa),
        )
        if self._physics_fields:
            args += (np.int32(vr.size), self._physics_device)
        if self._has_features:
            args += (*self._feature_static, self._feature_buffer,
                     np.int32(scale_index // 2 if graph_scales is not None else 0),
                     np.int32(self._n // self._spatial_n), np.int32(vr.size))
        args += self._noise_arguments(
            model, scale_index // 2 if graph_scales is not None else noise_step,
            graph_scales is not None)
        kernel(((self._n + 255) // 256,), (256,), args)

    def _graph_kernel(self, model):
        if not hasattr(self, "_graph_kernel_cache"):
            self._graph_kernel_cache = {}
        key = (model, self._batched)
        if key not in self._graph_kernel_cache:
            source = self._source(model, graph=True)
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
            self._zero_res = self.xp.zeros(shape, self._real_dtype)
            if self._has_features:
                self._prepare_features()
            self._graph_cache.clear()
            self._graph_signature = None
            self._pump_combiner = None
            if hasattr(self, "_zero_pump"):
                del self._zero_pump

    def advance(self, model, state, n_steps: int) -> None:
        if self._spinor:
            return super().advance(model, state, n_steps)
        """Run supported periodic chunks with planned FFTs and optional graphs."""
        potential = model.potential_static if model.time_dependent_potential else model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        start_step = state.step
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        if n_steps <= 0:
            return
        self._activate_shape(state)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.no_reservoir import NoReservoir

        models = {SingleReservoir: "single", DoubleReservoir: "double",
                  QuadraticDoubleReservoir: "quadratic-double", NoReservoir: "none"}
        reservoir_model = models.get(type(reservoir))
        diffusion = any(reservoir.batch_physics.any_nonzero(name) for name in
                        ("reservoir_diffusion_R", "reservoir_diffusion_A",
                         "reservoir_diffusion_I")) if reservoir_model not in (None, "none") else False
        absorption = getattr(boundary_condition, "absorption", None)
        if reservoir_model is None or diffusion or not getattr(absorption, "after_step_is_noop", False):
            return super().advance(model, state, n_steps)

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
                            length, reservoir_model, potential, envelope, pump_flat, combiner
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
                        if self._has_features:
                            pending_hosts.append(self._feature_values(
                                [state.t0 + (start_step + done + k) * self.config.solver.dt
                                 + self.config.solver.dt / 2 for k in range(length)], stream))
                        if self._noise is not None:
                            host_step = np.asarray([start_step + done], dtype=np.uint64)
                            pending_hosts.append(host_step)
                            self._noise_step_base.set(host_step, stream=stream)
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
                    if self._has_features:
                        pending_hosts.append(self._feature_values((t + self.config.solver.dt / 2,), stream))
                    self._launch_into(reservoir_model, local, res[parity], potential, pumps,
                                      factors, out, res[1 - parity],
                                      noise_step=start_step + k)
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
        state.step = start_step + n_steps
        state.t = state.t0 + state.step * self.config.solver.dt

    def step(self, model, state, dt):
        """Advance K(dt/2) [R(dt/2) N(dt) R(dt/2)] K(dt/2)."""
        if self._spinor:
            return self._step_spinor(model, state, dt)
        potential = model.potential_static if model.time_dependent_potential else model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        self._activate_shape(state)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.no_reservoir import NoReservoir

        psi_k = getattr(state, "field_k", None)
        if psi_k is None:
            psi_k = self._fft_forward(self.xp.ascontiguousarray(state.field), plan=self._plan(state.field))
        psi_k = psi_k.astype(self._complex_dtype, copy=False)
        psi = self._fft_inverse(psi_k * self._half_normalized, norm="forward",
                              plan=self._plan(psi_k))
        reservoir_state = reservoir.get_state()
        cls = type(reservoir)
        model = ({SingleReservoir: "single", DoubleReservoir: "double",
                  QuadraticDoubleReservoir: "quadratic-double", NoReservoir: "none"}).get(cls)
        diffusion = any(reservoir.batch_physics.any_nonzero(name) for name in
                        ("reservoir_diffusion_R", "reservoir_diffusion_A",
                         "reservoir_diffusion_I")) if model not in (None, "none") else False
        self._launch_time = state.t
        self._launch_step = state.step
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
            if self._noise is not None:
                self._noise.apply_arrays(
                    psi, reservoir.get_active_density(reservoir.get_state()), state.step)
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
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
