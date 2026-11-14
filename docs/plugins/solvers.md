# Solver plugins

A solver integrates the condensate and reservoir fields. The controller builds a
[`Model`](../api/model-components.md) containing the grid, batched physics,
potential, pump, reservoir, boundary condition, and optional stochastic noise.
A plugin receives this model at construction and for every step.

## Contract

```python
from polarism.solver import AbstractSolver, ACCEPTED, SolverCapabilities, register_solver

POLARISM_PLUGIN_API = 2

@register_solver("my-solver")
class MySolver(AbstractSolver):
    capabilities = SolverCapabilities(ndim={1, 2}, kinetic_relaxation=True,
                                      description="Example solver")

    def step(self, model, state, dt):
        rate, reservoir_rates = model.rhs(
            state.field, model.reservoir.get_state(), state.t
        )
        state.field = model.boundary.after_step_action(state.field + dt * rate)
        model.reservoir.set_state(tuple(
            field + dt * derivative
            for field, derivative in zip(model.reservoir.get_state(), reservoir_rates)
        ))
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
```

`AbstractSolver.__init__(config, grid, model)` sets `self.config`,
`self.physics`, `self.batch_physics`, and `self.xp`. The factory always calls
`SolverClass(config, grid, model)`. `model.physics` is `BatchPhysics`, whose
`get(name)` returns a scalar or an array broadcast across batch members.

`step(model, state, dt)` returns a frozen `StepReport`. Fixed-step solvers return
the shared `ACCEPTED` object. An accepted step updates `state.step` and sets
`state.t = state.t0 + state.step * dt`; this keeps stage times stable through
chunked runs and checkpoint resume. `state.field` and reservoir fields use
`(B, 1, *S)` arrays, with `S=(nx,)` or `(ny, nx)`. Assigning `state.field`
invalidates its spectral cache.

The inherited `advance(model, state, n_steps)` calls `step` and applies
`model.noise` after each accepted step. Override `advance` only when a solver
can combine multiple steps efficiently. The override owns the same time and
step updates, and must apply stochastic noise at the end of each completed
step. Pump samples come from `model.pump(t)` at the method's stage times.

`model.rhs(field, reservoir_fields, t, out=None)` is the finite difference
reference derivative. It includes the configured potential, interactions,
reservoir blueshifts and gain, decay, kinetic relaxation, and reservoir
rates. A simple plugin can use it directly. A solver with fused kernels can
read model components and use its own stage implementation.

## Capabilities

The controller checks `SolverClass.capabilities` during validation. The frozen
`SolverCapabilities` fields describe component counts, dimensions, grid
boundaries, built-in reservoirs, plugin reservoir support, potential, kinetic
relaxation, dispersion, stochastic noise, adaptive stepping, coherent drive,
quantitative reservoir support, and a description. Every field has a
conservative default (one component, 2D, periodic grid, all built-in
reservoirs through the generic interface, parabolic dispersion, both noise
modes, no kinetic relaxation), so declare only what differs. Set-valued
fields accept plain sets. Declare only modes that your solver actually
implements. A plugin with no declaration receives a
warning and conservative defaults: one component, 2D periodic grid, generic
reservoir interface, parabolic dispersion, no kinetic relaxation.

`custom_reservoirs` is `"quantitative"`, `"diagnostic"`, or `"unsupported"`.
The `quantitative_reservoirs` set identifies built-in reservoirs appropriate
for quantitative coupling comparisons. Built-in CUDA fused solvers currently
reject `reservoir_type: none`; the NumPy and CuPy solvers support it.

## Midpoint example

```python
--8<-- "examples/plugins/polarism_plugins/solver/midpoint_fdm.py"
```

The example evaluates `model.rhs` at the start and midpoint and updates the
condensate and reservoir together. Compare it with `rk4-fdm` using a small
step size when changing the model or solver.
