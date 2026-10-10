# Plan: evaluate the pump at solver stage times

## Problem

`SimulationController.run` evaluates the total pump once per step, at the step
start time `t`, and passes the frozen array to `solver.step`. Every solver then
uses `P(t)` in all of its stages. For a time-dependent pump (`pulse-gaussian`)
this reduces every solver to first order in time (measured order 1.01 for all
seven solvers; RK4-type solvers should give 4, ETD-RK2 2).

## Decision

`AbstractSolver.step` receives the pump as a callable `pump(t) -> array`
instead of an array. The step start time is `state.t` (the controller already
keeps it equal to `step * dt` before each step). This intentionally breaks the
signature for plugin solvers.

## Changes

1. `polarism/solver/abstract_solver.py`: `step(potential, pump, reservoir,
   boundary_condition, state)` where `pump: Callable[[float], array]`; the step
   advances `state` from `state.t` to `state.t + dt`. Update the docstring.
2. RK4-type solvers evaluate `p0 = pump(t)`, `p_half = pump(t + dt/2)`,
   `p1 = pump(t + dt)` once per step and use `p0` for stage 1, `p_half` for
   stages 2 and 3, `p1` for stage 4:
   - `rk4_fdm_solver.py`, `rk4_fdm_fused_solver.py`, `ip_rk4_solver.py`,
     `ifrk4_fft_cuda_solver.py`;
   - `rk4_cuda_solver.py`: all three CPU fallback paths and all three GPU
     paths (`_step_gpu`, `_step_gpu_double`, `_step_gpu_qdouble`): pass the
     per-stage pump buffer to each kernel launch (cast to precision and
     flattened, like the current single `pump_flat`). Kernels stay unchanged.
3. Solvers that advance the reservoir with `reservoir.step(dt, psi, pump_array)`
   (`etd_rk2_solver.py`, `split_step_fft_solver.py`) pass
   `pump(t + dt/2)`; the midpoint RK2 inside `reservoir.step` stays second
   order with the midpoint pump. The `AbstractReservoir` API does not change.
4. `simulation_controller.py`: pass `self._compute_total_pump` to
   `solver.step`; remove the per-step `P_total = self._compute_total_pump(t)`.
   Result output still evaluates the pump at `t_after`.
5. Update every direct caller of `solver.step` in `tests/` (pass a callable,
   e.g. `lambda t: pump_array`, and keep the test semantics unchanged).
6. Docs: `docs/api/solvers.md` (interface section) and
   `docs/user-guide/plugins.md` if it describes solver plugins.

## Out of scope

Split-step Strang splitting (separate fix), CUDA stream synchronisation,
pump caching, any other refactor.

## Acceptance criteria

- New fast test `tests/convergence/test_pump_time_order.py` (not marked slow,
  under ~15 s total): pulsed pump (`pulse-gaussian`, `sigma_time=0.1`,
  `cutoff_sigma=3`, pulse fully inside the run so the cutoff jumps fall on step
  boundaries), small periodic grid, active single reservoir and gain; temporal
  self-convergence order of `psi` and `nR` with three or more `dt` values:
  - `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda` (CPU path), `ip-rk4`,
    `ifrk4-fft-cuda` (CPU path): order >= 3.7;
  - `etd-rk2`: order >= 1.8.
  The test fails on the current code (order ~1).
- `pytest -q` (default selection) passes.
- `pytest -q -m '' tests/crosscheck tests/test_solver_compliance.py
  tests/test_solver_decay.py tests/test_solver_uniformity.py tests/reference
  tests/test_reservoir_0d_stationary.py` passes on CPU (GPU-only tests skip).
- `mkdocs build --strict` passes.
