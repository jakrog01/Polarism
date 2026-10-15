# Time Evolution

The codebase supports several real-time integration strategies. The best choice depends on accuracy needs, grid topology, and available hardware.

## Solver families

| Solver | Method key | Typical use |
| --- | --- | --- |
| Reference finite difference RK4 | `rk4-fdm` | Baseline correctness and convergence studies |
| Fused finite difference RK4 | `rk4-fdm-fused` | Faster CPU or backend-agnostic finite-difference runs |
| CUDA fused RK4 | `rk4-cuda` | GPU production runs |
| Split-step FFT | `split-step-fft` | Second-order Strang splitting on periodic grids |
| CUDA split-step FFT | `split-step-fft-cuda` | Second-order spectral evolution on a periodic CUDA grid |
| Interaction-picture RK4 | `ip-rk4` | Spectral-style evolution when its assumptions fit |
| ETD-RK2 | `etd-rk2` | Periodic spectral evolution with exponential differencing |
| GPU-native Lawson IF-RK | `ifrk-fft-cuda` | Selectable fourth-, fifth-, seventh-, eighth-, or ninth-order spectral runs |
| Lawson IF-RK4 alias | `ifrk4-fft-cuda` | Historical fourth-order method name |

## Choosing a solver

Recommended order:

1. Prototype with `rk4-fdm`.
2. Confirm time-step and grid convergence.
3. Move to `rk4-fdm-fused` or `rk4-cuda` (FDM production) once the physics setup is stable.
4. For periodic-grid spectral campaigns, use `ifrk-fft-cuda` (GPU-native with built-in reservoirs, stage-coupled).
5. Use `split-step-fft` or `split-step-fft-cuda` for second-order spectral evolution on periodic grids, including reservoir-coupled cases. Neither evaluates `kinetic_relaxation_eta`.

## Spectral solver limitations

All spectral solvers (`split-step-fft`, `split-step-fft-cuda`, `ip-rk4`, `etd-rk2`, `ifrk-fft-cuda`, `ifrk4-fft-cuda`)
are FFT-only and therefore accept the `periodic` grid only.  Selecting a
`closed-interval` grid raises `SolverCompatibilityError` at configuration time.
For `closed-interval` domains use the finite-difference family: `rk4-fdm`,
`rk4-fdm-fused` or `rk4-cuda`.

See [Running on GPUs](running-on-gpus.md) for CUDA solver requirements and
CUDA graph chunking.

`etd-rk2` advances the reservoir via a separate midpoint step, not stage-coupled
with `psi`.  It is not a production path for `quadratic-double` threshold studies.

## Time-step guidance

There is no hard-coded CFL guard in the runtime path. You must validate `dt` yourself.

Practical procedure:

1. Run the same case with progressively smaller `dt`.
2. Track observables such as maximum density, integrated norm, and threshold times.
3. Stop decreasing `dt` only when those observables are stable at the tolerance your study needs.

The bundled tests already encode this mindset by checking decay laws, spatial uniformity, and solver-to-solver agreement.

## Real time only

The current package does not expose a dedicated imaginary-time evolution driver. If you need true ground-state workflows, add them explicitly and document the numerical assumptions before using them for production research.

## Batched simulations

A leading axis runs independent simulations on one grid. Construct a state and
zero reservoir fields with the same `batch_size`:

```python
from polarism import SimulationState, create_reservoir

state = SimulationState(grid, seed=7, batch_size=3)
reservoir = create_reservoir(cfg.reservoir, cfg.physics, grid,
                             precision="double", batch_size=3)
```

`state.psi` and every reservoir field have shape `(3, ny, nx)`. The seed
produces the same initial wavefunction in every member as a single unbatched
state with that seed; assign `state.psi` to customize members. Pass one shared
`(ny, nx)` potential and a callable `pump(t)` returning either a shared
`(ny, nx)` pump or individual `(3, ny, nx)` pump fields. The boundary profile
and mask are shared across the batch. `reservoir.reset(batch_size=3)` restores
batched zero fields; `reservoir.reset()` restores unbatched zero fields.

All built-in solvers accept batched states. Spatial derivatives and FFTs use the
last two axes, so batch members do not interact. See the [solver batch
capability table](../api/solvers.md#batched-solver-capabilities) for CUDA paths.
