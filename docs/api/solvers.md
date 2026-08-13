# Solvers

All solver backends implement the `AbstractSolver` interface: they accept a configured system state and advance it by one timestep.

## Solver interface

At the abstraction level, every solver must:

- be constructible from the simulation config and grid
- implement `step(...)`
- consume the current potential, pump field, reservoir, boundary condition, and simulation state

## Available solver methods

| `solver.method` | Family | Intended role |
| --- | --- | --- |
| `rk4-fdm` | finite-difference RK4 | reference solver, broadest baseline |
| `rk4-fdm-fused` | optimized finite-difference RK4 | lower-overhead FDM path |
| `rk4-cuda` | fused CUDA RK4 | GPU-oriented production runs |
| `rk4-cuda-v100` | hardware-tuned CUDA RK4 | V100-specialized variant of the CUDA path |
| `split-step-fft` | spectral split-step | efficient spectral solver when assumptions match |
| `etd-rk2` | exponential time differencing | spectral periodic-grid path |
| `ip-rk4` | interaction-picture RK4 | spectral-style RK4 variant |
| `ifrk4-fft-cuda` | GPU-native FFT interaction-picture RK4 | production spectral solver with stage-coupled reservoirs |

## Compatibility guidance

The package includes explicit compatibility checks and warnings. In practice:

- `rk4-fdm` is the safest baseline when you need a reference answer
- `rk4-cuda` is the FDM production/reference path for GPU campaigns; supports periodic and closed-interval
- `ifrk4-fft-cuda` is the spectral production path: GPU-native cuFFT, periodic + CAP, quadratic-double stage-coupled
- `split-step-fft` and `ip-rk4` on `closed-interval` grids use SciPy DCT with host-device copies every step — CPU-bound on GPU runs; diagnostic use only
- `etd-rk2` and `ifrk4-fft-cuda` are restricted to periodic grids
- `etd-rk2` advances the reservoir between predictor and corrector using a midpoint condensate estimate, retaining second-order coupling for all supported reservoir types
- CUDA solvers are meaningful only when the GPU backend is actually active

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
  RK4 stages as `psi`, eliminating the O(dt) operator-splitting error present in
  split-step-based approaches.
- **No hard k-space filter**: high-k content is visible in `high_k_frac_0p8_nyq`
  diagnostics without artificial suppression.
- **Energy relaxation**: `kinetic_relaxation_eta != 0` adds
  `eta * nR * laplacian(psi)` to the nonlinear RHS.  This term has a spatially
  variable coefficient and is therefore evaluated in real space at every stage, not
  folded into the integrating factor.

## RK4 CUDA Laplacian Options

Both `rk4-cuda` and `rk4-cuda-v100` support a selectable finite-difference
Laplacian through `solver.laplacian`.

| `solver.laplacian` | Description | Notes |
| --- | --- | --- |
| `five-point` | nearest-neighbor 2D Laplacian | default, historical behavior |
| `isotropic-9pt` | 9-point isotropic stencil | requires square cells (`dx == dy`) |

Both CUDA solvers compile the same kernel stencil for the selected Laplacian;
`rk4-cuda-v100` is numerically identical to `rk4-cuda` and only differs in
block geometry and `__launch_bounds__` tuning for V100 occupancy.

The 9-point stencil is a numerical discretization of the same kinetic operator,
not a new physical term.  It is useful when high-k modes expose the angular
anisotropy of the five-point stencil.  It does not replace grid-convergence or
time-step checks.

For `quadratic-double` campaigns there are two production paths depending on
grid topology:

- **`rk4-cuda`** is the FDM production and reference path.  It supports both
  periodic and closed-interval grids and integrates `psi`, `nR`, `nI` in the same
  fused CUDA RK4 scheme.  Use it for quantitative threshold/amplitude studies,
  especially on closed-interval grids.
- **`ifrk4-fft-cuda`** is the spectral production path for periodic grids (or
  periodic + CAP open-boundary problems).  It stage-couples `(nR, nI)` inside the
  same FFT RK4 scheme as `psi`, eliminating the O(dt) splitting error.  Use it
  for large periodic-grid campaigns where spectral kinetics are preferred.
- **`split-step-fft`** is diagnostic only for `quadratic-double`. Its Lie
  splitting advances the reservoir with a separate RK2 step at the end of each
  full split-step cycle, introducing a global O(dt) coupling error. It is not a
  production reference for threshold or amplitude comparisons.

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
