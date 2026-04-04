# Grid and Parameters

The full runtime configuration is held in `polarism.config.simulation_parameters.Config`, a nested dataclass tree that separates numerical, physical, and output concerns.

## Main configuration sections

| Section | Purpose |
| --- | --- |
| `grid` | Spatial resolution, box size, and grid topology |
| `physics` | Material constants, interaction strengths, and decay rates |
| `boundary_condition` | Absorption type, profile, and mask width |
| `potential` | External potential family and parameters |
| `laser` | Pump definition for single or multiple sources |
| `reservoir` | Single or double reservoir model |
| `solver` | Time step, total time, precision, and solver method |
| `result` | Visualization and storage controls |
| `compute_engine` | CPU versus GPU backend selection |

## Grid configuration

The solver currently targets 2-D grids. The main controls are:

```python
cfg.grid.nx = 256
cfg.grid.ny = 256
cfg.grid.lx = 100.0
cfg.grid.ly = 100.0
cfg.grid.grid_type = "periodic"
```

Supported `grid_type` values:

- `periodic`
- `closed-interval`

Use `periodic` for FFT-based solvers and clean spectral behavior. Use `closed-interval` when you need finite-domain behavior, but prefer FDM-family solvers for the best agreement on that grid type.

## Physics constants

The `physics` block controls decay, interaction, and reservoir coupling parameters such as:

- `gamma_C`, `gamma_R`
- `g_C`, `g_R`
- `R`
- `m_eff`
- `hbar`
- `init_eps`

These values directly enter the nonlinear GPE-reservoir dynamics. Keep units consistent across the full configuration; the package does not perform unit conversion for you.

## Solver controls

At minimum, define:

```python
cfg.solver.method = "rk4-fdm"
cfg.solver.dt = 1e-3
cfg.solver.total_time = 5.0
cfg.solver.precision = "double"
```

Practical guidance:

- Start from `rk4-fdm` to establish a reference solution.
- Reduce `dt` until observables such as `|psi|^2`, norm growth, or threshold times stop moving materially.
- Increase `nx` and `ny` only after the time-step sensitivity is under control.

## Backend selection

The compute backend is explicit:

```python
cfg.compute_engine.use_gpu = True
cfg.compute_engine.gpu_device = 0
```

If CuPy is unavailable, the backend falls back to NumPy. That fallback is convenient for portability, but it is not a substitute for validating CUDA paths on the target hardware.
