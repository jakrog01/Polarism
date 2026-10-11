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
`ifrk-fft-cuda` samples once for each distinct stage time `t + c_i dt` in its
selected tableau.

`SimulationController` passes a `PumpField` (`polarism.laser.pump_field`).
Besides `pump(t)`, it offers `separable_term()`, which returns
`(envelope, coefficient)` when there is exactly one built-in laser. `rk4-cuda`
then scales the cached envelope inside its kernels instead of building three
pump arrays per step. Any other callable falls back to `pump(t)`.
`rk4-cuda` uses one tiled CUDA kernel per step in single precision and on GPUs
classified as full-rate FP64 in double precision. The tile includes a stencil halo
in shared memory. Other double-precision GPUs, and devices without enough shared
memory for a tile, use staged kernels. Tiled and staged results agree to rounding
level.

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
| `ifrk-fft-cuda` | selectable-order Lawson interaction-picture RK | periodic spectral solver with stage-coupled reservoirs |
| `ifrk4-fft-cuda` | Lawson RK4 alias | equivalent to `ifrk-fft-cuda` with `tableau: rk4` |

## Compatibility guidance

The package includes explicit compatibility checks and warnings. In practice:

- `rk4-fdm` is the safest baseline when you need a reference answer
- `rk4-cuda` is the FDM production/reference path for GPU campaigns; supports periodic and closed-interval
- `ifrk-fft-cuda` is the spectral production path: GPU-native cuFFT for built-in reservoirs, periodic + CAP, stage-coupled reservoirs
- all spectral solvers (`split-step-fft`, `split-step-fft-cuda`, `ip-rk4`, `etd-rk2`, `ifrk-fft-cuda`, `ifrk4-fft-cuda`) are FFT-only and restricted to periodic grids; a `closed-interval` grid raises `SolverCompatibilityError`
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

For CUDA graph eligibility, precision guidance, benchmarking, and device
selection, see [Running on GPUs](../user-guide/running-on-gpus.md).

## ifrk-fft-cuda

`ifrk-fft-cuda` advances the kinetic term exactly in Fourier space and applies
an explicit Runge–Kutta tableau to the nonlinear condensate and reservoir
terms. Choose the fixed-step tableau with `cfg.solver.tableau`. The default is
`rk4`; `ifrk4-fft-cuda` is a compatibility alias that always uses RK4.

| Tableau | Order | Stages | Embedded order | Coefficient source |
| --- | ---: | ---: | ---: | --- |
| `rk4` | 4 | 4 | — | Classical RK4 |
| `dp5` | 5 | 7 | 4 | SciPy RK45, Dormand–Prince 1980 |
| `vern7` | 7 | 10 | 6 | OrdinaryDiffEq.jl, Verner 2010 |
| `dop853` | 8 | 12 | 5 and 3 | SciPy DOP853, Prince–Dormand / Hairer–Nørsett–Wanner |
| `vern9` | 9 | 16 | 8 | OrdinaryDiffEq.jl, Verner 2010 |

The embedded weights are stored for future error estimation; this solver
currently takes fixed steps. Work per step grows roughly with the stage count
(two FFTs per stage, plus one inverse FFT per stage when energy relaxation is
enabled). Start with `rk4` for routine runs. Higher order methods can reduce
the timestep required for tight time accuracy on smooth solutions. `vern7` and
`vern9` are most useful in double precision: single precision typically
saturates around 1e-6 to 1e-7 relative accuracy. Consumer GPUs often have
slow double precision; V100, A100, and H100 have much stronger FP64 throughput.

### Accuracy versus cost

Measured with Polarism 0.3.5 on an RTX 4070 SUPER: 1024² periodic grid, double
precision, default physics and double reservoir, one `continuous-gaussian`
pump, 3.2 ps run, chunked execution without output. The error is the relative
L2 error of ψ against `ifrk-fft-cuda` with `vern9` at dt = 0.0125 ps. Wall
times include no compilation or graph capture. Double precision on this
consumer GPU is slow; on V100/A100/H100 the higher-order rows become relatively
cheaper.

| solver | tableau | dt [ps] | error | wall time |
| --- | --- | ---: | ---: | ---: |
| `split-step-fft-cuda` | — | 0.1 | 1.7e-7 | 0.05 s |
| `split-step-fft-cuda` | — | 0.025 | 1.1e-8 | 0.18 s |
| `split-step-fft-cuda` | — | 0.00625 | 6.8e-10 | 0.73 s |
| `ifrk-fft-cuda` | `rk4` | 0.2 | 1.5e-8 | 0.10 s |
| `ifrk-fft-cuda` | `rk4` | 0.1 | 2.5e-9 | 0.20 s |
| `ifrk-fft-cuda` | `rk4` | 0.05 | 6.2e-10 | 0.40 s |
| `ifrk-fft-cuda` | `dp5` | 0.2 | 5.4e-9 | 0.21 s |
| `ifrk-fft-cuda` | `dp5` | 0.1 | 1.4e-9 | 0.42 s |
| `ifrk-fft-cuda` | `vern7` | 0.4 | 8.3e-9 | 0.16 s |
| `ifrk-fft-cuda` | `vern7` | 0.2 | 2.1e-9 | 0.32 s |
| `ifrk-fft-cuda` | `vern7` | 0.1 | 5.5e-10 | 0.63 s |
| `ifrk-fft-cuda` | `vern9` | 0.2 | 1.1e-9 | 0.55 s |
| `ifrk-fft-cuda` | `vern9` | 0.1 | 3.5e-10 | 1.10 s |

In this run the errors of all tableaus fall only about two- to fourfold per
halving of dt. The cause is the initial condition, not the integrator: the
default `init_mode` seeds ψ with white noise that has content up to the grid's
Nyquist wavenumber, and Lawson methods lose their classical order on such
non-smooth data. With a smooth start (`init_mode =
"filtered_complex_gaussian"`, `init_k_cutoff_um = 0.5`, 256² grid, same
reference) the nominal orders appear:

| tableau | dt = 0.4 | dt = 0.2 | dt = 0.1 | dt = 0.05 |
| --- | ---: | ---: | ---: | ---: |
| `rk4` | 1.6e-9 | 4.3e-10 | 6.3e-11 | 4.3e-13 |
| `dp5` | 1.0e-9 | 1.6e-10 | 7.5e-12 | 1.3e-14 |
| `vern7` | 3.4e-10 | 2.7e-11 | 2.8e-13 | 8.2e-15 |
| `vern9` | 8.3e-11 | 2.3e-12 | 1.4e-14 | 8.1e-15 |

Practical guidance:

- `split-step-fft-cuda` is the fastest choice for tolerances around 1e-7 or
  looser.
- `ifrk-fft-cuda` with `rk4` is the most efficient choice for noise-seeded runs
  that need tighter time accuracy.
- `vern7` and `vern9` pay off for smooth solutions and tolerances near double
  precision, where they reach a given error with far larger steps. `dop853` was
  less accurate than `vern7` at every step size we measured.
- `rk4-cuda` uses finite differences, so its spatial error dominates any
  comparison against a spectral reference; it is unstable above dt ≈ 0.003 ps
  on this 1024² grid because the explicit limit scales with dx².

Key properties:

- **GPU-native**: built-in reservoirs without diffusion use fused CUDA kernels,
  reusable buffers, and planned cuFFT transforms. CPU runs and plugin reservoirs
  use the same Lawson stages through the generic NumPy/CuPy path.
- **Requires `grid_type: periodic`**: raises `ValueError` at construction for
  `closed-interval`.  Open-boundary problems should use a wide CAP absorber on a
  periodic grid.
- **Reservoirs are stage-coupled**: each reservoir field advances through the
  selected tableau's stages alongside `psi`.
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
