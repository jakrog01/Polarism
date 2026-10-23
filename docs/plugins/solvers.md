# Solver plugins

## When to write one

Write a solver when you need an integrator with different numerical stages,
boundary treatment, or performance behavior. The example here is a compact
explicit midpoint (second-order RK2) finite-difference method, implemented
without inheriting a built-in solver's `step`.

## Contract

`create_solver` calls `SolverClass(config, grid)`. Subclass `AbstractSolver`;
call `super().__init__(config)` to get `self.config`, `self.physics`, and
`self.xp`.

For per-member physics, the factory calls
`SolverClass(config, grid, batch_physics=values)`, where `values` maps a
`PhysicsConstants` float field to a one-dimensional array of length `B`.
Accept this optional constructor argument only if the solver supports a leading
batch axis on canonical `(B, 1, *S)` fields and applies each member's constants. The factory keeps
the two-argument call for ordinary runs and raises a named error if a physics
batch selects a plugin without `batch_physics`.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `__init__(config, grid)` | Factory, after grid and reservoir setup | Validate method constraints; retain grid data | Required |
| `step(potential, pump, reservoir, boundary_condition, state)` | Default `advance`, once per step | Update canonical `state.field` and reservoir; returns `None` | Abstract |
| `advance(potential, pump, reservoir, boundary_condition, state, start_step, n_steps)` | Controller, once per output chunk | Advance exactly `n_steps`; maintain stage times | Inherited; override only for chunking |
| `pump(t)` | Called by `step` at its stage times | Shared `grid.shape` or member-stacked `(B, 1, *S)` array | Input callable |
| `state.t` | Read by `step` | Start time; inherited `advance` updates it afterward | Input state |
| `state.field` / `state.field_k` | Read or written by `step` | Canonical real-space field / optional spectral cache, both `(B, 1, *S)` | State contract |

The solver decides when to sample `pump` and when to call
`reservoir.get_derivatives`, `reservoir.step`, or `set_state`. A real-space
method should pass the final field through
`boundary_condition.after_step_action(field)`. Assigning `state.field` clears its
`field_k` cache. A spectral method that keeps Fourier state must maintain
`state.field_k` and real-space materialization consistently.
Here `S` is the spatial shape `(nx,)` or `(ny, nx)`. FFT methods transform over
`tuple(range(-grid.ndim, 0))`; shared potential and pump arrays use `grid.shape`.

!!! warning "Plugin compatibility is your responsibility"
    `solver_compatibility.py` stores capabilities for built-in method names.
    An unknown plugin name defaults to 2D support and bypasses the other grid,
    reservoir, potential, and kinetic-relaxation checks. Validate unsupported settings in your own
    constructor and test the configurations you advertise.

## Step-by-step

1. Create `polarism_plugins/solver/midpoint_fdm.py` and register a unique
   solver name.
2. In `__init__`, call the base initializer and reject unsupported grids or
   physics terms. The example supports periodic grids and requires zero
   `kinetic_relaxation_eta`.
3. Write the condensate and reservoir right-hand sides using `self.xp` and
   the reservoir interface. The example computes a five-point periodic
   Laplacian with `xp.roll`; this private helper is specific to the example.
4. In `step`, sample `pump(state.t)` and `pump(state.t + dt/2)`, form a coupled
   midpoint state, then use its derivative for the full update.
5. Apply the boundary after-step action, assign `state.field`, and call
   `reservoir.set_state`. Leave the time update to inherited `advance`.
6. Compare with `rk4-fdm` on a small periodic case and reduce `dt` when
   checking accuracy for your own model.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/solver/midpoint_fdm.py"
```

The `_rhs` function reproduces the configured kinetic, potential,
interaction, reservoir blueshift, and gain/loss terms for this example's
supported settings. The first derivative predicts a midpoint `psi` and
reservoir tuple; the second advances both by `dt`. `state.field` is cast back
to its original complex dtype before storage.

## GPU and performance notes

`xp.roll` and the other `self.xp` operations run on NumPy or CuPy. This
teaching solver uses the inherited step loop and makes no CUDA graph; it is
not presented as a faster replacement for the built-in fused solvers. Avoid
host transfers in `_rhs`, and benchmark on the target hardware before making
performance claims.

## Supported by solvers

| Relationship | Behavior |
| --- | --- |
| Controller with `cfg.solver.method = "example-midpoint-fdm"` | Factory selects this plugin; controller calls its inherited `advance` |
| Built-in `rk4-fdm` | Reference for a short-run comparison, not a dependency of the plugin |
| Other built-in solvers | Alternative `method` selections, not called by this plugin |

The example calls the generic reservoir state and derivative methods, so it
can run with the built-in reservoirs and plugins that implement that contract.
Its own constructor limits it to periodic grids.

## Pitfalls

- `pump` is a callable, not the pump array for the current step.
- Do not increment `state.t` in `step` when using inherited `advance`.
- A solver registered under a built-in name inherits that name's compatibility
  checks even if its implementation differs.
- Preserve the configured precision and handle any optional model term you
  claim to support; reject unsupported terms explicitly.

## Testing

```python
candidate.solver.method = "example-midpoint-fdm"
midpoint = SimulationController(candidate)
midpoint.run()
reference.solver.method = "rk4-fdm"
rk4 = SimulationController(reference)
rk4.run()
np.testing.assert_allclose(midpoint.state.psi, rk4.state.psi,
                           rtol=1e-3, atol=1e-6)
```

The checked integration test also compares reservoir fields after five short
steps. The tolerance is for that test case, not a general error guarantee.
