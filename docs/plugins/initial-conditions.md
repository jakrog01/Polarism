# Initial-condition plugins

## When to write one

Write an initial condition for a deterministic start field that the built-in
`zero`, `plane-wave`, `vortex` and `file` types do not provide, such as a
Gaussian packet or a soliton. The controller evaluates it once per member
during setup and adds the result to the random field from `noise.initial`.

## Contract

There is no base class. `create_initial_condition` looks up the selected
function and calls it once per member.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `condition_fn(grid, initial_condition_config, xp, cdtype)` | `SimulationState`, once per member during setup | NumPy/CuPy array of shape `grid.shape` or `(2, *grid.shape)` | Required |
| `@register_initial_condition("name")` | Module import | Registers the function under `name` | Required |

The function returns one spatial field or a two-component spinor field. The framework adds the batch axis,
casts the result to `cdtype`, and rejects any other shape. In a spinor run, a single field is added to the plus component only. `grid.X` (and
`grid.Y` in 2D) hold the coordinates in µm, centred on the origin, and
`grid.ndim` is 1 or 2. `xp` is NumPy or CuPy, so no separate import is needed.
`InitialConditionParameters` holds `amplitude`, `x0`, `y0`, `kx`, `ky`,
`charge`, `core_width`, `path`, `frame` and an `extra` dictionary for
plugin-specific keys.

The built-in `zero` type is never evaluated: the start field is then exactly the
noise field, which keeps existing runs bit-identical.

## Step-by-step

1. Create `polarism_plugins/initial_condition/gaussian_packet.py`.
2. Define a function taking `grid`, `cfg`, `xp` and `cdtype`; decorate it with
   a unique `register_initial_condition` name.
3. Return a full field built from `grid.X` and `grid.Y`, and select the name
   with `cfg.initial_condition.type`.
4. Construct `SimulationController(cfg)` and inspect `controller.state.psi`
   before running.

## Complete example

```python
from polarism.initial_condition import register_initial_condition


@register_initial_condition("gaussian-packet")
def gaussian_packet(grid, cfg, xp, cdtype):
    radius_squared = (grid.X - cfg.x0) ** 2
    if grid.ndim == 2:
        radius_squared = radius_squared + (grid.Y - cfg.y0) ** 2
    envelope = xp.exp(-radius_squared / cfg.extra.get("width", 5.0) ** 2)
    return cfg.amplitude * envelope * xp.exp(1j * cfg.kx * grid.X)
```

```python
import numpy as np
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = cfg.grid.ny = 8
cfg.grid.lx = cfg.grid.ly = 16.0
cfg.solver.dt = 0.001
cfg.solver.total_time = 0.002
cfg.initial_condition.type = "gaussian-packet"
cfg.initial_condition.amplitude = 0.5
cfg.initial_condition.extra = {"width": 8.0}
```

## GPU and performance notes

Use operations on `grid.X` and `grid.Y` so the result stays on their backend.
Evaluation happens once per member before the run, so it adds nothing to the
time-stepping loop.

## Sweeps

`initial_condition.type` and `initial_condition.path` form separate sweep
groups. Every other field, including `extra.<key>` for a key already present,
can vary inside one batch.

## Pitfalls

- Return `grid.shape` for a scalar field or `(2, *grid.shape)` for a spinor field. A scalar value or any other shape raises an error.
- The returned field is added to the noise field. Set `noise.enabled = False`
  for a purely deterministic start.
- The vortex type needs a 2D grid; a plugin that does should check `grid.ndim`.

## Testing

```python
import numpy as np
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = cfg.grid.ny = 8
cfg.grid.lx = cfg.grid.ly = 16.0
cfg.solver.dt = 0.001
cfg.solver.total_time = 0.002
cfg.initial_condition.type = "gaussian-packet"
cfg.noise.enabled = False
run = ps.SimulationController(cfg)
assert run.state.psi.shape == run.grid.shape
assert np.isfinite(run.state.psi).all()
```
