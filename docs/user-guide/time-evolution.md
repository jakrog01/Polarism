# Time Evolution

The codebase supports several real-time integration strategies. The best choice depends on accuracy needs, grid topology, and available hardware.

## Solver families

| Solver | Method key | Typical use |
| --- | --- | --- |
| Reference finite difference RK4 | `rk4-fdm` | Baseline correctness and convergence studies |
| Fused finite difference RK4 | `rk4-fdm-fused` | Faster CPU or backend-agnostic finite-difference runs |
| CUDA fused RK4 | `rk4-cuda` | GPU production runs |
| V100-tuned CUDA RK4 | `rk4-cuda-v100` | GPU runs specialized for V100-class hardware |
| Split-step FFT | `split-step-fft` | Spectral runs on simple periodic-like setups |
| Interaction-picture RK4 | `ip-rk4` | Spectral-style evolution when its assumptions fit |
| ETD-RK2 | `etd-rk2` | Periodic spectral evolution with exponential differencing |
| GPU-native FFT IP-RK4 | `ifrk4-fft-cuda` | GPU-native spectral production runs with stage-coupled reservoirs |

## Choosing a solver

Recommended order:

1. Prototype with `rk4-fdm`.
2. Confirm time-step and grid convergence.
3. Move to `rk4-fdm-fused` or `rk4-cuda` (FDM production) once the physics setup is stable.
4. For periodic-grid spectral campaigns, use `ifrk4-fft-cuda` (GPU-native, stage-coupled quadratic-double).
5. Use `split-step-fft`, `ip-rk4`, or `etd-rk2` only as diagnostic or reference spectral paths.

## Spectral solver limitations

`split-step-fft` and `ip-rk4` on `closed-interval` grids invoke SciPy DCT and
perform host-device copies on every timestep.  On a GPU backend this is CPU-bound
and orders of magnitude slower than `ifrk4-fft-cuda`.  Both solvers emit a
`UserWarning` when this path is active.

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
