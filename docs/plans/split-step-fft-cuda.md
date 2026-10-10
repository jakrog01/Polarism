# Plan: `split-step-fft-cuda` (0.3.1)

Goal: a fast, second-order, large-time-step GPU solver for periodic grids.
Measured on 256² (T = 12.8 ps): the existing CuPy `split-step-fft` is stable
at dt = 0.256 ps (rel. error 1.4e-4), where `rk4-cuda` is unstable above
~0.05 ps; Strang order is 1.99 for single and quadratic-double reservoirs
(today's convergence run).

## Splitting: K-outer Strang, two FFTs per step

`split-step-fft` uses R(dt/2) K(dt/2) N(dt) K(dt/2) R(dt/2). R needs |psi|²
in real space after each K half, so it costs 4 transforms per step.

The new solver uses the equally symmetric Strang composition

    K(dt/2) · [ R(dt/2) N(dt) R(dt/2) ] · K(dt/2)

where R is the reservoir half-step at fixed psi (pump at t + dt/4 and
t + 3dt/4) and N the exact local condensate step (same formula as
`split-step-fft`). The bracket is purely local in real space, so one step is

    psi_k --[K(dt/2) · 1/N]--> ifft --[fused R N R]--> fft --[K(dt/2)]--> psi_k

i.e. **2 FFTs + 3 pointwise kernels per step** (two cheap k-space
multiplies, one fused real-space kernel). The state stays in
`state.psi_k` between steps, like `ifrk4-fft-cuda`; `state.psi` is
materialized by one ifft only when results are recorded. psi_k at a step
boundary is always the true state (no lagged half-step), so recording,
restarts and other consumers need no special handling.

Second order, same physics as `split-step-fft`, different error constant
(results agree with `split-step-fft` to O(dt²), not bitwise). A NumPy
reference implementation of exactly this composition lives in the tests for
the bitwise-level crosscheck.

## Scope

- New `polarism/solver/split_step_fft_cuda_solver.py`, registered as
  `split-step-fft-cuda`; periodic grid only; precision single/double, no
  float64 leaks (propagators cast to the solver dtype).
- Reservoirs: built-in `single`, `double`, `quadratic-double` in the fused
  kernel (same equations as their `get_derivatives`, RK2 midpoint per half,
  clamp at 0 like `set_state`). Nonzero reservoir diffusion: the R halves
  need a stencil, so they run as separate stencil kernels (periodic
  Laplacian) — correctness first, fast path for zero diffusion.
  Custom/plugin reservoirs: fall back to `reservoir.step(dt/2, psi, pump)`
  around a fused N kernel.
- Pump: `PumpField.separable_term()` (envelope × per-half scalar inside the
  kernel, rounded product as in rk4-cuda); otherwise `pump(t)` arrays.
- Potential: real part → phase, imaginary part → gain (CAP), as in
  `split-step-fft`; real potentials pass `V_re` twice with scale 0 (no zero
  stream).
- cuFFT via `cupyx.scipy.fft` plans (cached, in place where possible); the
  ifft normalization is folded into the K multiply.
- After-step absorption that is not a no-op (`mask`): materialize, apply,
  transform back (2 extra FFTs) — documented as slower; `cap` is the fast
  path.
- `kinetic_relaxation_eta` unsupported (same as `split-step-fft`), declared
  in `solver_compatibility.py`.
- CPU (no CuPy): not supported by this solver; compatibility check says so
  (use `split-step-fft`).

## Tests

- Crosscheck vs a NumPy reference of the same K-outer composition: rel L2
  < 1e-10 (double) / 1e-5 (single), 3 reservoirs, CAP and zero potential,
  pulsed pump, one and two lasers (separable and array pump paths).
- Agreement with `split-step-fft` at small dt (O(dt²) difference shrinks
  ×4 when dt halves).
- `test_time_convergence`: order 2 for single and quadratic-double.
- Determinism (bitwise), dtype preservation in single precision, mask
  absorption path.
- `scripts/analyse_convergence.py`: fix the stale split-step band for
  quadratic-double ([0.5, 1.5] -> [1.5, 2.5]) and add the new solver.

## Benchmark / docs

- `scripts/bench_gpu_solvers.py`: add the solver (FFT lower-bound model: 2
  transforms).
- Error–time comparison (split-step-fft-cuda vs ifrk4-fft-cuda vs rk4-cuda)
  at matched accuracy, published in `docs/api/solvers.md` with guidance on
  when to use it.

## Acceptance

- All tests above pass; full `pytest -m '' --use-gpu` once before release.
- Target: ≥ 4x faster per step than `ifrk4-fft-cuda` (8 FFT) on 1024²/2048²;
  ≥ 10x faster than the CuPy `split-step-fft`.
