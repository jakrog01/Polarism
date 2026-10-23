# Reservoir plugins

## When to write one

Write a reservoir when you need a new rate equation or state layout beyond
the built-in single, double, and quadratic-double models. A reservoir supplies
densities to the condensate equation and evolves one or more real fields.

## Contract

`create_reservoir` calls
`ReservoirClass(reservoir_config, physics, grid, precision)`. Subclass
`AbstractReservoir`. Its abstract constructor declaration omits `precision`,
but a concrete plugin must accept it because the factory passes it.

For per-member physics, the factory calls
`ReservoirClass(reservoir_config, physics, grid, precision,
batch_physics=values)`. `values` maps a `PhysicsConstants` float field to a
one-dimensional array of length `B`. A plugin that accepts it must apply each
member's constants to the leading axis of canonical `(B, 1, *S)` arrays. Ordinary runs keep the existing
four-argument call; the factory raises a named error when a physics batch
selects a plugin without `batch_physics`.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `__init__(reservoir_config, physics, grid, precision)` | Factory, once | Initialize real `(1, 1, *S)` fields and coefficients | Required |
| `get_state()` | Stage-coupled solvers and `step` | Tuple of canonical `(B, 1, *S)` arrays | Abstract |
| `set_state(state)` | Solvers after a step | Store a tuple matching `get_state()` | Abstract |
| `get_derivatives(psi, pump, state, rho=None)` | RK4 stages and example `step` | Canonical `psi`, `state`, `rho`; derivatives match state shape | Abstract |
| `step(dt, psi_mid, pump)` | Split-step and midpoint-style solvers | Update stored fields; returns `None` | Abstract |
| `get_reservoir_density()` | Client code | Public `grid.shape` or `(B, *S)` array | Abstract |
| `make_result_nodes()` | Controller output setup | List of `ResultNode`; `[]` is valid | Abstract |
| `get_active_density(state)` | Solvers at stage evaluations | Density coupled to condensate; defaults to `state[0]` | Optional override |
| `get_inactive_density(state)` | Solvers at stage evaluations | Inactive density; defaults to zero array | Optional override |

!!! note "Tuple order is part of the contract"
    `get_derivatives()` must return one derivative for each field in
    `get_state()`, in the same order. Built-in stage-coupled solvers combine
    these tuples element by element.

`S` is `(nx,)` in 1D or `(ny, nx)` in 2D. Diffusion follows the existing axes:
periodic wrapping or mirrored Neumann boundaries on each spatial axis.

## Step-by-step

1. Create `polarism_plugins/reservoir/slow_single.py` and register a new name.
2. Initialize backend arrays with `compute_engine.xp` and
   `real_dtype(xp, precision)`; read custom coefficients from
   `reservoir_config.extra`.
3. Implement state access and the rate equation. Use the passed `state`, not
   only stored fields, in `get_derivatives` so RK stages work correctly.
4. Implement `step` for solvers that call the reservoir directly. Return
   result nodes, or `[]` if this model exposes no reservoir-specific output.
5. Select `cfg.reservoir.reservoir_type` and test with both `rk4-fdm` and
   `split-step-fft` if users need both integration families.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/reservoir/slow_single.py"
```

The example implements `AbstractReservoir` directly. `decay_factor` multiplies
`physics.gamma_R`; `1.0` reproduces the built-in single-reservoir rate, while
smaller positive values slow decay. `get_derivatives` accepts the optional
precomputed `rho` and includes the configured reservoir diffusion term.
`step` uses an explicit midpoint update. `set_state` clips negative density
and preserves the configured real dtype. This teaching example returns no
reservoir result nodes, though the condensate and pump nodes still work.

## GPU and performance notes

Keep all state and derivative arrays on `self.xp`; avoid host conversions in
`get_derivatives` and `step`. The generic solvers can call the plugin on CuPy
arrays. `split-step-fft-cuda` uses regular stepping for an unrecognized
reservoir class; its fused chunk and graph path recognize exact built-in
reservoir classes. Compare small CPU and GPU runs before production use.

## Supported by solvers

| Solver | Plugin reservoir behavior |
| --- | --- |
| `rk4-fdm`, `rk4-fdm-fused`, `ip-rk4`, `etd-rk2`, `ifrk-fft-cuda`, `ifrk4-fft-cuda` | Compatibility metadata accepts custom reservoirs; call their derivative/state interface |
| `split-step-fft`, `split-step-fft-cuda` | Compatibility metadata accepts custom reservoirs; call `step` in regular stepping |
| `rk4-cuda` | Compatibility check rejects custom `reservoir_type` |

## Pitfalls

- A subclass of a built-in reservoir still counts as a plugin for
  `split-step-fft-cuda`'s exact-class fast-path check.
- Changing the tuple order without updating density accessors changes the
  condensate coupling.
- `make_result_nodes()` is called only when result nodes are built; do not
  rely on it for physics initialization.

## Testing

```python
cfg.reservoir.reservoir_type = "example-slow-single"
cfg.reservoir.extra = {"decay_factor": 1.0}
run = SimulationController(cfg)
run.run()
assert np.isfinite(run.reservoir.get_reservoir_density()).all()
```

The [integration test](https://github.com/jakrog01/Polarism/blob/master/tests/integration/test_example_plugins.py)
compares the neutral plugin with the built-in `single` reservoir under both
`rk4-fdm` and `split-step-fft`.
