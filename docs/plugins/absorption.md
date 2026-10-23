# Absorption strategy plugins

## When to write one

Write an absorption strategy for a new boundary damping profile or a
potential-based absorbing term. The strategy is wrapped by
`BoundaryCondition`, which exposes before-step and after-step actions to
solvers.

## Contract

`create_absorption_strategy` calls
`StrategyClass(grid, boundary_config, physics)`. Subclass
`AbsorptionStrategy`. Its abstract constructor declaration takes no
arguments, but the factory passes those three objects.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `__init__(grid, boundary_config, physics)` | Factory, once | Store profile and parameters | Required |
| `set_time_step(dt)` | `BoundaryCondition`, after construction | Store `dt` if per-step damping needs it | Optional override |
| `get_potential_distribution()` | Controller setup, once | Scalar or `grid.shape` potential term | Abstract |
| `apply_absorption(psi)` | Solver after a step | Updated real-space field with trailing `grid.shape` | Abstract |
| `after_step_is_noop` | CUDA chunk eligibility | True only if `apply_absorption` leaves `psi` unchanged | Optional class attribute |

`create_absorption_profile(grid.shape, boundary_config)` builds a configured
edge profile. It measures distance from both x walls in 1D and from all x and
y walls in 2D. A strategy can use that helper or construct another profile.

## Step-by-step

1. Create `polarism_plugins/absorption/soft_edge.py` and register a name.
2. In `__init__`, compute a profile with grid shape and save the configured
   strength. Use `compute_engine.xp` for backend operations.
3. Store `dt` in `set_time_step`; `BoundaryCondition` calls it before the
   simulation runs.
4. Return zero from `get_potential_distribution` for a pure after-step mask.
   Implement `apply_absorption` to multiply the completed `psi` by the mask.
5. Select `cfg.boundary_condition.absorption` and compare edge and center
   values in a small test.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/absorption/soft_edge.py"
```

`create_absorption_profile` supplies edge weights. The example returns zero
potential, then applies `exp(-strength * dt * profile / 2)` after each step.
It sets `after_step_is_noop = False` because that multiplication changes the
field. The `physics_constants` argument is accepted because the factory
passes it, though this particular mask does not use it.

## GPU and performance notes

The profile and mask calculation use `compute_engine.xp`, so they remain on
CuPy for GPU runs. Both `rk4-cuda` and `split-step-fft-cuda` require an
after-step no-op for optimized chunks and CUDA graphs. A mask like this takes
their regular stepping path. For a CAP-style strategy, return the complex
potential in `get_potential_distribution`, leave `psi` unchanged in
`apply_absorption`, and then set `after_step_is_noop = True`.

## Supported by solvers

| Solver family | Plugin strategy behavior |
| --- | --- |
| `rk4-fdm`, `rk4-fdm-fused`, `split-step-fft`, `etd-rk2`, `ip-rk4`, `ifrk-fft-cuda`, `ifrk4-fft-cuda` | Use the combined potential and after-step action |
| `rk4-cuda`, `split-step-fft-cuda` | Accept the strategy; post-step actions disable their optimized chunk path |

## Pitfalls

- Do not apply the same damping in both the potential and mask paths unless
  that double action is intentional.
- `after_step_is_noop=True` must reflect `apply_absorption` behavior, not just
  the absence of a Python exception.
- A post-step mask is a separate operation after the integrator's stages.

## Testing

```python
cfg.boundary_condition.absorption = "example-soft-edge"
run = SimulationController(cfg)
field = np.ones(run.grid.shape, dtype=np.complex128)
damped = run.boundary_condition.after_step_action(field)
assert np.abs(damped).max() <= 1.0
```

The integration suite additionally runs two complete steps with this strategy.
