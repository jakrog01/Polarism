# Solvers and time stepping

Choose an integrator that supports your grid and model, then test time step convergence for the quantities you intend to report.

## Choose a finite difference solver

```yaml
solver:
  method: rk4-fdm
  dt: 0.01
```

Times and time steps are in ps. `rk4-fdm` is the reference finite difference method and works with periodic and closed-interval grids. For a larger run, choose from these families:

| Method | Grid | Typical use |
| --- | --- | --- |
| `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda` | Periodic or closed-interval | Finite difference RK4, including CUDA production runs |
| `split-step-fft`, `split-step-fft-cuda` | Periodic | Second-order spectral splitting |
| `ip-rk4`, `etd-rk2` | Periodic | CPU spectral integration |
| `ifrk-fft-cuda` | Periodic | GPU spectral integration with selectable tableau |

`ifrk4-fft-cuda` is an alias for the fourth-order CUDA interaction-frame method. The full compatibility and accuracy matrix is in [Solver reference](../reference/solvers.md). GPU setup is in [Running on GPUs](running-on-gpus.md).

## Choose a time step

For explicit finite difference solvers, validation warns when `dt` exceeds the kinetic or reservoir diffusion stability estimate, but it does not stop the run. Run representative cases at `dt`, `dt/2`, and `dt/4` ps. Compare density, norm, threshold times, and any spectral quantity used in the study. Refine the spatial grid as well. A stable-looking image alone is not a convergence check.

`solver.total_time` is the target time from the initial state, in ps. Scheduled output, checkpoints, events, and accumulator samples use multiples of `solver.dt`. The initial state is sampled at its starting time. See [Configuration reference](../reference/configuration.md) for all solver settings and defaults.

## Adaptive integration

On CPU and CUDA, `ifrk-fft-cuda` supports adaptive steps with the `dp5`, `dop853`, `vern7`, and `vern9` tableaus. Set `solver.adaptive: true` and choose relative and absolute error tolerances. `solver.dt` remains the schedule tick and initial trial step, in ps. Internal steps stop at scheduled boundaries and may otherwise span several ticks. `solver.dt_max` caps an internal step in ps.

Adaptive integration compares condensate and reservoir errors. Batched members share the step selected by the largest error. It uses the generic NumPy or CuPy path, without the fused CUDA kernels or CUDA graphs, and synchronizes with the host once per trial step. Stochastic Wigner noise and mask absorption are unsupported in adaptive mode, and single precision needs `rtol` of at least `1e-5`. Defaults are `rtol: 1e-6` and `atol: 1e-8`. `controller.step_statistics` reports accepted and rejected trials.

## Stop a run early

```yaml
solver:
  dt: 0.01
  total_time: 300.0
stop:
  steady_state: true
  check_every: 10.0
  rtol: 1.0e-6
  min_time: 50.0
```

`stop.check_every` and `stop.min_time` are in ps. Each check compares the norm and density with the preceding check. Both relative changes must be below `stop.rtol`. A norm below `1e-30` is treated as steady. A batch continues until all members stop or reach `total_time`; recorded stop times are in ps.

For a custom condition, register `controller.add_event(callback, every=...)` before `run()`. `every` is in ps and must be a positive multiple of `solver.dt`. The callback receives a read-only host view with `t`, `step`, `psi`, `density`, `reservoir`, `batched`, and `batch_size`. Return `True` to stop all members, or a boolean array to stop selected members. Register callbacks again after loading a checkpoint.

## Batched states

A batch advances independent members on the same grid. `SimulationState(..., batch_size=B)` and `create_reservoir(..., batch_size=B)` create a leading member axis; the canonical field shape is `(B, C, *S)`, where `C` is one or two components and `S` is the spatial shape. Built-in solvers accept batched states. [Parameter sweeps](parameter-sweeps.md) construct and group batches for you.

Polarism evolves in real time. It does not provide an imaginary-time ground-state driver.
