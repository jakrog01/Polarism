# Solvers

All solver backends implement the `AbstractSolver` interface: they accept a configured system state and advance it by one timestep.

## Solver interface

At the abstraction level, every solver must:

- be constructible from the simulation config and grid
- implement `step(potential, pump, reservoir, boundary_condition, state)`
- consume the current potential, pump callable, reservoir, boundary condition, and simulation state

`pump` is a callable `pump(t) -> array`, returning the pump field at the
requested time. The step starts at `state.t` and advances the fields by
`config.solver.dt`; the caller updates `state.t` afterward.

RK4 solvers evaluate the pump once at each of `t`, `t + dt/2`, and `t + dt`,
using the midpoint value for both middle stages. `etd-rk2` passes
`pump(t + dt/2)` to `reservoir.step`. `split-step-fft` advances the reservoir
in two half-steps, sampling the pump at `t + dt/4` and `t + 3dt/4`. The
reservoir interface still accepts an array. For a constant field, pass
`lambda t: pump_array`. Treat the returned array as read-only.

`SimulationController` passes a `PumpField` (`polarism.laser.pump_field`).
Besides `pump(t)`, it offers `separable_term()`, which returns
`(envelope, coefficient)` when there is exactly one built-in laser. `rk4-cuda`
then scales the cached envelope inside its kernels instead of building three
pump arrays per step. Any other callable falls back to `pump(t)`.

This replaces the former pump-array argument and intentionally breaks the
plugin solver interface. Plugin solvers must evaluate the callable at their
integration stage times.

## Available solver methods

| `solver.method` | Family | Intended role |
| --- | --- | --- |
| `rk4-fdm` | finite-difference RK4 | reference solver, broadest baseline |
| `rk4-fdm-fused` | optimized finite-difference RK4 | lower-overhead FDM path |
| `rk4-cuda` | fused CUDA RK4 | GPU-oriented production runs |
| `split-step-fft` | spectral Strang split-step | second-order periodic-grid solver |
| `split-step-fft-cuda` | two-transform CUDA spectral Strang split-step | fast large-dt second-order runs on periodic grids |
| `etd-rk2` | exponential time differencing | spectral periodic-grid path |
| `ip-rk4` | interaction-picture RK4 | spectral-style RK4 variant |
| `ifrk4-fft-cuda` | GPU-native FFT interaction-picture RK4 | production spectral solver with stage-coupled reservoirs |

## Compatibility guidance

The package includes explicit compatibility checks and warnings. In practice:

- `rk4-fdm` is the safest baseline when you need a reference answer
- `rk4-cuda` is the FDM production/reference path for GPU campaigns; supports periodic and closed-interval
- `ifrk4-fft-cuda` is the spectral production path: GPU-native cuFFT, periodic + CAP, quadratic-double stage-coupled
- all spectral solvers (`split-step-fft`, `split-step-fft-cuda`, `ip-rk4`, `etd-rk2`, `ifrk4-fft-cuda`) are FFT-only and restricted to periodic grids; a `closed-interval` grid raises `SolverCompatibilityError`
- for `closed-interval` grids use the FDM family: `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda`
- `etd-rk2` advances the reservoir between predictor and corrector using a midpoint condensate estimate, retaining second-order coupling for all supported reservoir types
- CUDA solvers are meaningful only when the GPU backend is actually active

`split-step-fft-cuda` keeps the wavefunction in Fourier space between steps and
uses two cuFFT transforms per step. Its local kernel combines two reservoir RK2
half-steps with the exact condensate step. It supports all built-in reservoirs,
including reservoir diffusion via a slower periodic stencil path, and custom
reservoirs through their `step` method. A mask absorber costs two additional
transforms; CAP absorption stays on the fast path. It requires an active CUDA
GPU and does not support `kinetic_relaxation_eta`. Like `split-step-fft`, it is
second order, so compare timestep errors before using large steps.

Measured on an RTX 4070 SUPER: 1024² periodic grid, double precision, 3.2 ps
run. The error is the relative L2 error of ψ against a fine-dt `ifrk4-fft-cuda`
reference.

| solver | dt [ps] | time error | wall time |
| --- | ---: | ---: | ---: |
| `rk4-cuda` (largest stable dt) | 0.002 | 8e-8 | 1.66 s |
| `split-step-fft-cuda` | 0.064 | 1.4e-5 | 0.08 s |
| `split-step-fft-cuda` | 0.016 | 8.9e-7 | 0.31 s |
| `ifrk4-fft-cuda` | 0.128 | 8.7e-7 | 0.19 s |
| `ifrk4-fft-cuda` | 0.064 | 5.4e-8 | 0.37 s |

`rk4-cuda` is unstable above dt ≈ 0.003 ps on this grid; the explicit limit
scales with dx². Both spectral solvers treat the kinetic term exactly.
`split-step-fft-cuda` is the fastest choice for exploratory runs with
tolerances around 1e-6 or looser. Use `ifrk4-fft-cuda` for tighter time
accuracy.

## ifrk4-fft-cuda

`ifrk4-fft-cuda` is a GPU-native spectral solver with no CPU involvement in the
time loop.

Key properties:

- **GPU-native**: all transforms use `xp.fft.fft2` / `xp.fft.ifft2` (cuFFT when CuPy
  is active); there are no `.get()`, `.asnumpy()`, or `numpy.asarray()` calls inside
  `step()`.
- **Requires `grid_type: periodic`**: raises `ValueError` at construction for
  `closed-interval`.  Open-boundary problems should use a wide CAP absorber on a
  periodic grid.
- **Quadratic-double is stage-coupled**: `(nR, nI)` are advanced through the same four
  RK4 stages as `psi`.
- **No hard k-space filter**: high-k content is visible in `high_k_frac_0p8_nyq`
  diagnostics without artificial suppression.
- **Energy relaxation**: `kinetic_relaxation_eta != 0` adds
  `eta * nR * laplacian(psi)` to the nonlinear RHS.  This term has a spatially
  variable coefficient and is therefore evaluated in real space at every stage, not
  folded into the integrating factor.

## RK4 CUDA Laplacian Options

`rk4-cuda` supports a selectable finite-difference Laplacian through
`solver.laplacian`.

| `solver.laplacian` | Description | Notes |
| --- | --- | --- |
| `five-point` | nearest-neighbor 2D Laplacian | default, historical behavior |
| `isotropic-9pt` | 9-point isotropic stencil | requires square cells (`dx == dy`) |

The 9-point stencil is a numerical discretization of the same kinetic operator,
not a new physical term.  It is useful when high-k modes expose the angular
anisotropy of the five-point stencil.  It does not replace grid-convergence or
time-step checks.

For `quadratic-double` campaigns, choose a solver based on grid topology and
the desired time integrator:

- **`rk4-cuda`** is the FDM production and reference path.  It supports both
  periodic and closed-interval grids and integrates `psi`, `nR`, `nI` in the same
  fused CUDA RK4 scheme.  Use it for quantitative threshold/amplitude studies,
  especially on closed-interval grids.
- **`ifrk4-fft-cuda`** is the spectral production path for periodic grids (or
  periodic + CAP open-boundary problems).  It stage-couples `(nR, nI)` inside the
  same FFT RK4 scheme as `psi`. Use it
  for large periodic-grid campaigns where spectral kinetics are preferred.
- **`split-step-fft`** uses second-order Strang splitting for all built-in
  reservoir types, including `quadratic-double`. Two reservoir half-steps
  surround the kinetic and exact local condensate steps. It supports a
  non-zero potential and CAP absorption on periodic grids, but does not
  evaluate `kinetic_relaxation_eta`.

`etd-rk2` inserts its reservoir sub-step between the predictor and corrector and
drives it with the midpoint condensate estimate `(psi_n + a) / 2`. This retains
O(dt²) psi-reservoir coupling for every listed reservoir type, including
`quadratic-double` (measured p≈2.00 in the convergence study).

## Selection strategy

Use solver choice as a numerical decision, not just a performance switch:

1. establish a stable baseline with `rk4-fdm`
2. check timestep and grid convergence
3. compare against a faster solver on representative cases
4. only then move to production throughput runs
