# Grid plugins

## When to write one

Write a grid plugin to supply different coordinates or grid metadata while keeping one of Polarism's supported boundary semantics. Subclass a built-in grid when its coordinates and Fourier frequencies suit your model.

## Contract

`create_grid` calls `GridClass(grid_config)`. The `SimulationGrid` protocol describes the shared spatial attributes.

| Name | Meaning |
| --- | --- |
| `boundary` class attribute | `"periodic"` or `"closed-interval"`; checked by `register_grid` and solver compatibility |
| `ndim` | Spatial dimension, 1 or 2 |
| `shape` | Spatial array shape: `(nx,)` or `(ny, nx)` |
| `spacing`, `lengths` | Tuples in the same axis order as `shape`: `(dx,)` or `(dy, dx)`, `(lx,)` or `(ly, lx)` |
| `nx`, `lx`, `dx`, `X`, `kx`, `KX` | x-axis size, length, spacing, coordinates, frequencies, and spectral coordinates |
| `grid_type`, `k_squared` | Registry identifier and spectral field of shape `grid.shape` |
| `ny`, `ly`, `dy`, `Y`, `ky`, `KY` | Required only for 2D grids; absent from 1D grids |

`X`, `KX`, and `k_squared` use shape `grid.shape`. In 2D, `X`, `Y`, `KX`, and `KY` use `(ny, nx)`. A 1D plugin should leave y-axis attributes undefined so accidental 2D assumptions fail clearly.

The decorator rejects classes without a valid `boundary`. Solver compatibility uses that class attribute rather than the registration name. `GridParameters.extra` can carry setup values.

## Step-by-step

1. Create `polarism_plugins/grid/labelled_periodic.py`.
2. Subclass `PeriodicGrid` and register a distinct name.
3. Call the base constructor to fill coordinates and frequencies, then read custom metadata from `cfg.extra`.
4. Set `cfg.grid.grid_type = "example-labelled-periodic"` and construct a controller. Check `grid.shape` and `grid.spacing` before running.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/grid/labelled_periodic.py"
```

The inherited constructor supports both dimensions. The class declares periodic boundary semantics and adds a `label` from `GridParameters.extra`.

## GPU and performance notes

The built-in base uses `compute_engine.xp`, so inherited arrays are NumPy on CPU or CuPy on an active GPU. A from-scratch grid should follow that rule and avoid moving coordinate arrays to the host during solver setup.

## Supported by solvers

| Boundary value | Built-in methods accepted by compatibility checks |
| --- | --- |
| `periodic` | All built-in methods |
| `closed-interval` | `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda` |

All built-in methods support `ndim` 1 and 2. The compatibility check treats a plugin solver without an `ndim` capability as 2D-only.

## Testing

```python
cfg.grid.ndim = 1
cfg.grid.grid_type = "example-labelled-periodic"
cfg.grid.extra = {"label": "trial"}
run = SimulationController(cfg)
assert run.grid.label == "trial"
assert run.grid.X.shape == run.grid.shape == (cfg.grid.nx,)
assert not hasattr(run.grid, "Y")
```
