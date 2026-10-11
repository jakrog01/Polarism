"""Fused CUDA stages and captured fixed-step chunks for Lawson RK."""
from __future__ import annotations

import numpy as np


_SOURCE = r"""
typedef {real} real;
#define MODEL {model}
#define ETA {eta}
#define STAGES {stages}
#define FIELDS {fields}
__device__ __forceinline__ void phase(real theta, real* sn, real* cs) {{
    {sincos}(theta, sn, cs);
}}
extern "C" __global__ void combine_k(
    const real* psi0, const real* f, const real* k2,
    const real* a, const real* b, const real* c,
    real* out, real* lap, int n, int stage, real dt, real phase_rate) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    real ci = stage == STAGES ? (real)1 : c[stage];
    real sn, cs;
    phase(phase_rate * dt * ci * k2[k], &sn, &cs);
    real re = cs * psi0[2*k] - sn * psi0[2*k+1];
    real im = sn * psi0[2*k] + cs * psi0[2*k+1];
    for (int j = 0; j < stage; ++j) {{
        real weight = dt * (stage == STAGES ? b[j] : a[stage*STAGES+j]);
        if (weight == (real)0) continue;
        phase(phase_rate * dt * (ci-c[j]) * k2[k], &sn, &cs);
        real fjre = f[2*(j*n+k)], fjim = f[2*(j*n+k)+1];
        re += weight * (cs*fjre - sn*fjim);
        im += weight * (sn*fjre + cs*fjim);
    }}
    real normalization = stage == STAGES ? (real)1 : (real)1 / (real)n;
    re *= normalization;
    im *= normalization;
    out[2*k] = re;
    out[2*k+1] = im;
#if ETA
    if (stage != STAGES) {{
        lap[2*k] = -k2[k] * re;
        lap[2*k+1] = -k2[k] * im;
    }}
#endif
}}
extern "C" __global__ void stage_rhs(
    const real* psi, const real* lap,
    const real* res0, const real* dr, const real* a,
    const real* vr, const real* vi, const real* pump,
    const real* scales, int scale_index,
    real* rhs, real* out_dr, int n, int stage, real dt,
    real inv_hbar, real hbar_over_2m, real gC, real gR, real gI,
    real R, real gammaC, real gammaR, real gammaI, real gammaA,
    real ria, real rai, real kappa, real eta) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    real active = res0[k];
    real inactive = MODEL == 0 ? (real)0 : res0[n+k];
    for (int j = 0; j < stage; ++j) {{
        real weight = dt * a[stage*STAGES+j];
        if (weight == (real)0) continue;
        active += weight * dr[(j*FIELDS)*n+k];
#if MODEL != 0
        inactive += weight * dr[(j*FIELDS+1)*n+k];
#endif
    }}
    real pre = psi[2*k], pim = psi[2*k+1];
    real rho = pre*pre + pim*pim;
    real energy = vr[k] + gC*rho + gR*active + gI*inactive;
    real gain = ((R*active-gammaC)*(real)0.5 + vi[k]*inv_hbar);
    real out_re = gain*pre + energy*inv_hbar*pim;
    real out_im = gain*pim - energy*inv_hbar*pre;
#if ETA
    real relax = eta*active*hbar_over_2m;
    out_re += relax*lap[2*k];
    out_im += relax*lap[2*k+1];
#endif
    rhs[2*k] = out_re;
    rhs[2*k+1] = out_im;
    real source = scales[scale_index] * pump[k];
#if MODEL == 0
    out_dr[(stage*FIELDS)*n+k] = source - (gammaR+R*rho)*active;
#elif MODEL == 1
    out_dr[(stage*FIELDS)*n+k] = ria*inactive - (gammaA+rai+R*rho)*active;
    out_dr[(stage*FIELDS+1)*n+k] = source - (gammaI+ria)*inactive + rai*active;
#else
    real transfer = kappa*inactive*inactive;
    out_dr[(stage*FIELDS)*n+k] = transfer - (gammaR+R*rho)*active;
    out_dr[(stage*FIELDS+1)*n+k] = source - transfer - gammaI*inactive;
#endif
}}
extern "C" __global__ void finish_res(
    const real* res0, const real* dr, const real* b,
    real* out, int n, real dt) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n*FIELDS) return;
    int field = k/n, point = k%n;
    real val = res0[k];
    for (int j = 0; j < STAGES; ++j) {{
        if (b[j] != (real)0) val += dt*b[j]*dr[(j*FIELDS+field)*n+point];
    }}
    out[k] = val < (real)0 ? (real)0 : val;
}}
"""


class FusedIFRK:
    """Own fixed tableau coefficients, stage storage, and cuFFT plans."""

    def __init__(self, solver, grid):
        from cupyx.scipy import fft

        self.solver = solver
        self.xp = solver.xp
        self.fft = fft
        self.shape = (grid.ny, grid.nx)
        self.n = grid.nx * grid.ny
        self.s = solver.tableau.stages
        self.real = solver._real_dtype
        self.complex = solver._complex_dtype
        self.scalar = solver._scalar
        self.k2 = self.xp.ascontiguousarray(solver._k_squared).ravel()
        self.a = self.xp.asarray(solver._a, dtype=self.real).ravel()
        self.b = self.xp.asarray(solver._b, dtype=self.real)
        self.c = self.xp.asarray(solver._c, dtype=self.real)
        self.unique_c = tuple(dict.fromkeys(solver._c))
        self.c_index = tuple(self.unique_c.index(c) for c in solver._c)
        self.dt = self.scalar(solver.config.solver.dt)
        self.rate = self.scalar(solver._phase_rate)
        self.eta = solver._eta != 0.0
        self.p = solver.config.physics
        self._kernels = {}
        self._plan = None
        self._buffers = None
        self._stream = self.xp.cuda.Stream(non_blocking=True)
        self._graph_cache = {}
        self._graph_signature = None
        self._graphs_failed = False
        self._potential_id = None
        self._zero_pump = self.xp.zeros(self.shape, dtype=self.real)

    def _model(self, reservoir):
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir

        models = {SingleReservoir: (0, 1), DoubleReservoir: (1, 2),
                  QuadraticDoubleReservoir: (2, 2)}
        if type(reservoir) not in models:
            return None
        if any(getattr(reservoir, name, 0.0) != 0.0 for name in
               ("reservoir_diffusion_R", "reservoir_diffusion_A", "reservoir_diffusion_I")):
            return None
        return models[type(reservoir)]

    def eligible(self, reservoir):
        return self._model(reservoir) is not None

    def _array(self, value, dtype):
        return self.xp.ascontiguousarray(self.xp.asarray(value, dtype=dtype))

    def _potential(self, potential):
        if id(potential) != self._potential_id:
            self.vr = self._array(self.xp.real(potential), self.real).ravel()
            self.vi = (self._array(self.xp.imag(potential), self.real).ravel()
                       if self.xp.iscomplexobj(potential) else self._zero_pump.ravel())
            self._potential_id = id(potential)
            self._graph_cache.clear()

    def _ensure(self, model):
        xp = self.xp
        from cupy.cuda import cufft
        model_id, fields = model
        if self._buffers is not None and self._model_id == model_id:
            return
        self._model_id = model_id
        spec = tuple(xp.empty(self.shape, self.complex) for _ in range(2))
        res = tuple(xp.empty((fields, *self.shape), self.real) for _ in range(2))
        f = xp.empty((self.s, *self.shape), self.complex)
        dr = xp.empty((self.s, fields, *self.shape), self.real)
        stage_k = xp.empty(self.shape, self.complex)
        stage_psi = xp.empty_like(stage_k)
        lap_k = xp.empty_like(stage_k)
        lap_psi = xp.empty_like(stage_k)
        rhs = xp.empty_like(stage_k)
        self._buffers = (spec, res, f, dr, stage_k, stage_psi, lap_k, lap_psi, rhs)
        source = _SOURCE.format(real="float" if self.real == xp.float32 else "double",
                                model=model_id, eta=int(self.eta), stages=self.s,
                                fields=fields,
                                sincos="sincosf" if self.real == xp.float32 else "sincos")
        module = xp.RawModule(code=source, options=("--std=c++11",))
        self._kernels = {name: module.get_function(name)
                         for name in ("combine_k", "stage_rhs", "finish_res")}
        self._plan = self.fft.get_fft_plan(stage_k, axes=(-2, -1))
        self._graph_cache.clear()
        self._graph_scales = xp.empty((64, len(self.unique_c)), self.real)
        # Warm cuFFT, kernels, and the stream before graph capture.
        with self._stream:
            self._plan.fft(stage_k, stage_psi, cufft.CUFFT_INVERSE)
            self._plan.fft(stage_psi, f[0], cufft.CUFFT_FORWARD)
            self._stream.synchronize()

    def _pump(self, pump, t):
        term = pump.separable_term() if hasattr(pump, "separable_term") else None
        no_lasers = not getattr(pump, "lasers", [True])
        if term is not None:
            envelope, coefficient = term
            field = self._array(envelope, self.real).ravel()
            scales = [coefficient(t + c * float(self.dt)) for c in self.unique_c]
            return [field] * self.s, scales, field, coefficient
        if no_lasers:
            return ([self._zero_pump.ravel()] * self.s,
                    [1] * len(self.unique_c), self._zero_pump.ravel(), lambda time: 1)
        samples = [self._array(pump(t + c * float(self.dt)), self.real).ravel()
                   for c in self.unique_c]
        return [samples[i] for i in self.c_index], [1] * len(samples), None, None

    def _run_into(self, spectrum_in, res_in, spectrum_out, res_out,
                  pumps, scales, scale_base):
        xp = self.xp
        from cupy.cuda import cufft

        _, _, f, dr, stage_k, stage_psi, lap_k, lap_psi, rhs = self._buffers
        launch = ((self.n + 255) // 256,)
        block = (256,)
        scalar = self.scalar
        p = self.p
        for i in range(self.s):
            self._kernels["combine_k"](launch, block, (
                spectrum_in.view(self.real).ravel(), f.view(self.real).ravel(), self.k2,
                self.a, self.b, self.c, stage_k.view(self.real).ravel(),
                lap_k.view(self.real).ravel(), np.int32(self.n), np.int32(i),
                self.dt, self.rate))
            self._plan.fft(stage_k, stage_psi, cufft.CUFFT_INVERSE)
            if self.eta:
                self._plan.fft(lap_k, lap_psi, cufft.CUFFT_INVERSE)
            self._kernels["stage_rhs"](launch, block, (
                stage_psi.view(self.real).ravel(), lap_psi.view(self.real).ravel(),
                res_in.ravel(), dr.ravel(), self.a, self.vr, self.vi, pumps[i],
                scales, np.int32(scale_base + self.c_index[i]),
                rhs.view(self.real).ravel(), dr.ravel(), np.int32(self.n),
                np.int32(i), self.dt,
                scalar(1 / p.hbar), scalar(p.hbar / (2 * p.m_eff)),
                scalar(p.g_C), scalar(p.g_R), scalar(p.g_I), scalar(p.R),
                scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
                scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI),
                scalar(p.kappa), scalar(p.kinetic_relaxation_eta)))
            self._plan.fft(rhs, f[i], cufft.CUFFT_FORWARD)
        self._kernels["combine_k"](launch, block, (
            spectrum_in.view(self.real).ravel(), f.view(self.real).ravel(), self.k2,
            self.a, self.b, self.c, spectrum_out.view(self.real).ravel(),
            lap_k.view(self.real).ravel(), np.int32(self.n), np.int32(self.s),
            self.dt, self.rate))
        self._kernels["finish_res"](((self.n*res_in.shape[0]+255)//256,), block, (
            res_in.ravel(), dr.ravel(), self.b, res_out.ravel(),
            np.int32(self.n), self.dt))

    def _copy_in(self, state, reservoir):
        xp = self.xp
        spec, res = self._buffers[:2]
        spectrum = state.psi_k
        if spectrum is None:
            psi = self._array(state.psi, self.complex)
            spectrum = self.fft.fft2(psi, plan=self._plan)
        xp.copyto(spec[0], spectrum)
        for target, source in zip(res[0], reservoir.get_state()):
            xp.copyto(target, source)

    def _copy_out(self, index, reservoir, state):
        spec, res = self._buffers[:2]
        state.psi = None
        state.psi_k = spec[index].copy()
        reservoir.set_state(tuple(field.copy() for field in res[index]))

    def step(self, potential, pump, reservoir, boundary_condition, state):
        model = self._model(reservoir)
        self._ensure(model)
        self._potential(potential)
        pumps, factors, _, _ = self._pump(pump, state.t)
        scales = self.xp.asarray(factors, dtype=self.real)
        self._copy_in(state, reservoir)
        spec, res = self._buffers[:2]
        self._run_into(spec[0], res[0], spec[1], res[1], pumps, scales, 0)
        if not self.solver._can_skip_after_step(boundary_condition):
            state.psi = None
            state.psi_k = spec[1].copy()
            state.psi = boundary_condition.after_step_action(state.psi)
            state.psi_k = self.fft.fft2(state.psi, plan=self._plan)
            reservoir.set_state(tuple(field.copy() for field in res[1]))
        else:
            self._copy_out(1, reservoir, state)

    def _graph(self, length, signature, pump_flat):
        if signature != self._graph_signature:
            self._graph_cache.clear()
            self._graph_signature = signature
        if length in self._graph_cache:
            return self._graph_cache[length]
        stream = self._stream
        spec, res = self._buffers[:2]
        pumps = [pump_flat] * self.s
        stream.synchronize()
        with stream:
            stream.begin_capture()
            try:
                for k in range(length):
                    self._run_into(spec[k % 2], res[k % 2],
                                   spec[(k + 1) % 2], res[(k + 1) % 2],
                                   pumps, self._graph_scales.ravel(),
                                   k * len(self.unique_c))
                graph = stream.end_capture()
            except Exception:
                try:
                    stream.end_capture()
                except Exception:
                    pass
                raise
        if len(self._graph_cache) >= 4:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[length] = graph
        return graph

    def advance(self, potential, pump, reservoir, boundary_condition, state,
                start_step, n_steps):
        model = self._model(reservoir)
        self._ensure(model)
        self._potential(potential)
        xp = self.xp
        xp.cuda.get_current_stream().synchronize()
        spec, res = self._buffers[:2]
        stream = self._stream
        done = 0
        with stream:
            self._copy_in(state, reservoir)
            pumps, _, pump_flat, coefficient = self._pump(pump, state.t)
            graph_ok = (self.solver.config.solver.cuda_graphs and not self._graphs_failed
                        and coefficient is not None
                        and self.solver._can_skip_after_step(boundary_condition))
            if graph_ok:
                nodes_per_step = (4 + int(self.eta))*self.s + 2
                max_length = max(2, min(64, (1800 // nodes_per_step) // 2 * 2))
                signature = (model, id(potential), int(pump_flat.data.ptr))
                pending_hosts = []
                try:
                    while n_steps - done >= 2:
                        length = min(max_length, (n_steps-done)//2*2)
                        graph = self._graph(length, signature, pump_flat)
                        host = self.scalar([
                            coefficient((start_step+done+k)*float(self.dt)+c*float(self.dt))
                            for k in range(length) for c in self.unique_c
                        ]).reshape(length, len(self.unique_c))
                        pending_hosts.append(host)
                        self._graph_scales[:length].set(host, stream=stream)
                        graph.launch(stream=stream)
                        done += length
                        if len(pending_hosts) == 32:
                            stream.synchronize()
                            pending_hosts.clear()
                except Exception as exc:
                    import sys
                    self._graphs_failed = True
                    self._graph_cache.clear()
                    print(f"ifrk-fft-cuda: CUDA graphs unavailable ({exc}); using regular launches.",
                          file=sys.stderr)
                    stream.synchronize()
                    self._copy_in(state, reservoir)
                    done = 0
            for k in range(done, n_steps):
                t = (start_step+k)*float(self.dt)
                stage_pumps, factors, _, _ = self._pump(pump, t)
                scales = xp.asarray(factors, dtype=self.real)
                parity = (k-done) % 2
                self._run_into(spec[parity], res[parity], spec[1-parity],
                               res[1-parity], stage_pumps, scales, 0)
            index = (n_steps-done) % 2
            if self.solver._can_skip_after_step(boundary_condition):
                self._copy_out(index, reservoir, state)
            else:
                state.psi = None
                state.psi_k = spec[index].copy()
                state.psi = boundary_condition.after_step_action(state.psi)
                state.psi_k = self.fft.fft2(state.psi, plan=self._plan)
                reservoir.set_state(tuple(field.copy() for field in res[index]))
            stream.synchronize()
        state.t = (start_step+n_steps)*float(self.dt)
