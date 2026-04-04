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
- spectral solvers require more care when external potentials or closed-interval grids are involved
- `etd-rk2` is restricted to periodic grids
- CUDA solvers are meaningful only when the GPU backend is actually active

## Selection strategy

Use solver choice as a numerical decision, not just a performance switch:

1. establish a stable baseline with `rk4-fdm`
2. check timestep and grid convergence
3. compare against a faster solver on representative cases
4. only then move to production throughput runs
