# Potential plugins

## When to write one

Write a potential function for a spatial energy landscape that the built-in
zero and double-well models do not provide. It is evaluated during controller
setup, then combined with any boundary potential before solver construction.

## Contract

There is no potential base class. `create_potential` looks up the selected
function and calls it once.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `potential_fn(X, Y, potential_config)` | `create_potential`, once during setup | NumPy/CuPy array of shape `grid.shape` | Required |
| `@register_potential("name")` | Module import | Registers the function under `name` | Required |

`X` and `Y` already use the active backend. The controller calls `.astype` on
the potential in single precision, so a plain Python scalar is insufficient.
For a 1D grid, both arrays have shape `(nx,)` and `Y` is all zeros. The plugin
therefore evaluates its profile along y = 0. In 2D they have shape `(ny, nx)`.
`PotentialParameters` has fields such as `V1`, `V2`, positions, widths, and
order; it has no `extra` dictionary.

## Step-by-step

1. Create `polarism_plugins/potential/parabolic_trap.py`.
2. Define a function taking `X`, `Y`, and `potential_config`; decorate it
   with a unique `register_potential` name.
3. Use the supplied coordinate arrays, return a full field, and select the
   name with `cfg.potential.potential_type`.
4. Construct `SimulationController(cfg)` and inspect `controller.potential`
   before running; it includes the boundary potential as well.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/potential/parabolic_trap.py"
```

`V1` is the curvature. `X**2 + Y**2` gives an array with the grid's shape,
and the function does not need a separate NumPy or CuPy import.

## GPU and performance notes

Use operations on `X` and `Y` so the result stays on their backend. Potential
construction occurs once, but the solver uses the field repeatedly. For
`single` precision, the controller casts the final potential to float32 or
complex64. A complex potential is permitted by the controller; its imaginary
part affects gain or damping, so check its sign carefully.

## Supported by solvers

All eight registered built-in solver methods have `supports_potential=True`
in `solver_compatibility.py`. A plugin solver has no automatic capability
check, so test it with your potential explicitly.

## Pitfalls

- Do not return a Python float such as `0.0`; return an array.
- Do not assume a custom `extra` field exists on `PotentialParameters`.
- Resolve the sign and units of `V1` for your physical model.

## Testing

```python
cfg.potential.potential_type = "example-parabolic-trap"
cfg.potential.V1 = 0.001
run = SimulationController(cfg)
assert run.potential.shape == run.grid.shape
assert np.isfinite(run.potential).all()
```

The integration suite also runs this plugin through a short CPU simulation.
