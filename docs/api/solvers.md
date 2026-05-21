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

## Compatibility guidance

The package includes explicit compatibility checks and warnings. In practice:

- `rk4-fdm` is the safest baseline when you need a reference answer
- `rk4-cuda` is the production path for large coupled GPE/reservoir campaigns
- spectral solvers require more care when external potentials or closed-interval grids are involved
- `etd-rk2` is restricted to periodic grids
- CUDA solvers are meaningful only when the GPU backend is actually active

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

For `quadratic-double`, use `rk4-cuda` for quantitative threshold/amplitude
studies.  The split-step FFT path can be useful as a geometry diagnostic, but its
operator splitting is not the production reference for coupled `psi`, `nR`, `nI`
threshold measurements.

## Selection strategy

Use solver choice as a numerical decision, not just a performance switch:

1. establish a stable baseline with `rk4-fdm`
2. check timestep and grid convergence
3. compare against a faster solver on representative cases
4. only then move to production throughput runs
