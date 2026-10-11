# Grid plugins

## When to write one

Write a grid plugin to supply different coordinates or grid metadata while
keeping one of Polarism's supported boundary semantics. For a first plugin,
subclass a built-in grid so the coordinate and frequency arrays remain
consistent with existing solvers.

## Contract

`create_grid` calls `GridClass(grid_config)`. The `SimulationGrid2D` protocol
describes the attributes used by the controller and solvers.

| Name and signature | Caller and timing | Return or state | Status |
| --- | --- | --- | --- |
| `boundary` class attribute | `register_grid` and compatibility check | `"periodic"` or `"closed-interval"` | Required |
| `__init__(grid_config)` | `create_grid`, once | Populate all protocol attributes | Required |
| `nx`, `ny`, `lx`, `ly`, `dx`, `dy`, `grid_type` | Controller and solvers | Resolution, lengths, spacing, identifier | Required protocol fields |
| `X`, `Y` | Potential and laser factories | Backend coordinate arrays `(ny, nx)` | Required |
| `kx`, `ky`, `KX`, `KY`, `k_squared` | Spectral solvers | Frequency vectors and `(ny, nx)` spectral fields | Required protocol fields |

The decorator rejects classes without a valid `boundary`. Solver
compatibility uses that class attribute rather than the registration name.
`GridParameters.extra` can carry extra setup values.

## Step-by-step

1. Create `polarism_plugins/grid/labelled_periodic.py`.
2. Subclass `PeriodicSimulationGrid2D` and register a distinct name.
3. Call the base constructor to fill coordinates and frequencies, then read
   custom metadata from `cfg.extra`.
4. Set `cfg.grid.grid_type = "example-labelled-periodic"` and construct a
   controller. Check the shape and spacing before running.

## Complete example

```python
--8<-- "examples/plugins/polarism_plugins/grid/labelled_periodic.py"
```

The inherited constructor computes `dx`, `dy`, `X`, `Y`, and spectral arrays.
The class declares periodic boundary semantics and adds only a `label` from
`GridParameters.extra`.

## GPU and performance notes

The built-in base uses `compute_engine.xp`, so inherited arrays are NumPy on
CPU or CuPy on an active GPU. A from-scratch grid should follow that rule and
avoid moving coordinate arrays to the host during solver setup. Grid creation
occurs once, but every solver step depends on the correctness of its spacing.

## Supported by solvers

| Boundary value | Built-in methods accepted by compatibility checks |
| --- | --- |
| `periodic` | All eight built-in methods (CUDA methods also need a working GPU) |
| `closed-interval` | `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda` |

Declaring a boundary value does not itself prove a custom grid implements the
same numerical edges. Test the field and spectral arrays on rectangular as
well as square grids.

## Pitfalls

- `X` and `Y` use `(ny, nx)`; swapping axes may pass on square grids.
- FFT methods rely on consistent frequency arrays and periodic semantics.
- A subclass's inherited `grid_type` may still say `"periodic"`; the
  compatibility check uses `cfg.grid.grid_type` to find the registered class
  and then reads its `boundary` attribute.

## Testing

```python
cfg.grid.grid_type = "example-labelled-periodic"
cfg.grid.extra = {"label": "trial"}
run = SimulationController(cfg)
assert run.grid.label == "trial"
assert run.grid.X.shape == (cfg.grid.ny, cfg.grid.nx)
```

The integration suite also advances a short CPU run with this grid.
