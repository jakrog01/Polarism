# Plan: second-order Strang splitting in `split-step-fft`

## Problem

`split-step-fft` is first order in time in every case with gain or loss
(measured order 1.00 even with only `gamma_C > 0` and no reservoir coupling).
Its nominal order is 2. There are two independent causes:

1. The nonlinear sub-step multiplies `psi` by
   `exp((-i*eff/hbar + G) * dt)` with `eff = V + g_C*rho0 + ...`, where
   `rho0 = |psi|^2` is frozen at the start of the sub-step. With net gain/loss
   `G != 0` the density changes during the sub-step as `rho0*exp(2*G*s)`, so
   the phase has an O(dt^2) local error.
2. The reservoir is advanced once, after the full condensate step (Lie
   splitting of psi and reservoir).

## Splitting

Split the vector field into three exactly or second-order solvable flows:

- `K`: kinetic term, `psi_k *= exp(-i*hbar*k^2*h/(2*m_eff))` (exact).
- `N`: local condensate term with the reservoir frozen:
  `dpsi/dt = (-i*(V_re + g_C*|psi|^2 + g_R*nR + g_I*nI)/hbar + G) * psi`,
  `G = (R*nR - gamma_C)/2 + V_im/hbar`. Here `V` may be complex because the CAP
  is added to the potential: `-i*V/hbar = -i*V_re/hbar + V_im/hbar`. The exact
  solution over `h`, with `rho0 = |psi0|^2`, is
  `psi(h) = psi0 * exp(G*h) * exp(-i*[(V_re + g_R*nR + g_I*nI)*h + g_C*rho0*F]/hbar)`
  with `F = (exp(2*G*h) - 1)/(2*G)`. Compute `F` stably as
  `h * expm1(x)/x` with `x = 2*G*h`, using the limit `F = h` (or the series
  `h*(1 + x/2 + x^2/6)`) where `|x|` is tiny. Use `xp.real`/`xp.imag` so a real
  potential still works.
- `Rv`: reservoir with `psi` frozen, advanced by the existing
  `reservoir.step(h, psi, pump_array)` (midpoint RK2, second order). Time
  advances only inside this flow, so pass the pump at the midpoint of each
  reservoir sub-interval.

One step from `t` to `t + dt`:

```
Rv(dt/2) on [t, t+dt/2]          pump(t + dt/4)
K(dt/2)
N(dt)                            nR, nI read after the first Rv half-step
K(dt/2)
Rv(dt/2) on [t+dt/2, t+dt]       pump(t + 3*dt/4)
boundary_condition.after_step_action(psi)   (unchanged, last)
```

The composition is symmetric and every sub-flow is at least second order, so
the scheme is second order. `kinetic_relaxation_eta` stays unsupported.

## Changes

1. `polarism/solver/split_step_fft_solver.py`: implement the step above. Keep
   the precomputed half-step kinetic propagator. No other solver changes.
2. `polarism/solver/solver_compatibility.py`: the `split-step-fft` entry is now
   symmetric-split. Set `supports_potential` to True,
   `reservoir_types_quantitative` to all built-in reservoir types and
   `custom_reservoirs` to `"quantitative"`. Rewrite its description: Strang
   splitting, second order in time, reservoir in two half-steps, FFT-only,
   periodic only, no `kinetic_relaxation_eta`. Remove the unused
   `_LIE_SPLIT_RESERVOIR_TYPES` constant. Keep the generic warning machinery.
3. Tests that assert the old warnings for `split-step-fft` (potential warning,
   "Lie splitting" for `quadratic-double` and plugin reservoirs in
   `tests/unit/test_config_validation.py` and `tests/unit/test_plugins.py`)
   must now assert that no such warning is raised. Keep the other assertions
   in those tests.
4. Docs that describe `split-step-fft` as Lie-split, first order or diagnostic
   only: `docs/api/solvers.md`, `docs/api/option-matrix.md`,
   `docs/user-guide/plugins.md`, `docs/user-guide/time-evolution.md`. Describe
   it as second order (Strang). Keep the `eta` limitation.
5. Add `("split-step-fft", 1.8)` to the parametrization in
   `tests/convergence/test_pump_time_order.py`.
6. New fast test `tests/convergence/test_split_step_order.py` (not slow, a few
   seconds): temporal self-convergence order of `psi` and the first reservoir
   field for `split-step-fft`, three `dt` values, small periodic grid,
   with gain/loss on, for these cases:
   - `gamma_C > 0`, no reservoir coupling (`R = 0`, `g_R = 0`, zero pump);
   - `single` reservoir with a continuous pump;
   - `quadratic-double` reservoir;
   - `single` reservoir with `cap` absorption and a non-zero potential.
   Expect order >= 1.8. The test must fail on the current code (order ~1).

## Out of scope

Other solvers, ETD-RK2 phi thresholds, mask absorption, the slow convergence
suite, refactors.

## Acceptance criteria

- New and updated tests pass; the new tests fail on the original code.
- `pytest -q` (default selection) passes.
- `pytest -q -m '' tests/crosscheck tests/test_solver_compliance.py
  tests/test_solver_decay.py tests/test_solver_uniformity.py tests/reference
  tests/test_reservoir_0d_stationary.py tests/convergence/test_pump_time_order.py
  tests/convergence/test_split_step_order.py` passes on CPU.
- `mkdocs build --strict` passes.
