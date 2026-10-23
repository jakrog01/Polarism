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
| `reservoir` | Single, double, or quadratic-double reservoir model |
| `solver` | Time step, total time, precision, and solver method |
| `result` | Visualization and storage controls |
| `compute_engine` | CPU versus GPU backend selection |

## Grid configuration

The solver supports 1D and 2D grids. The main controls for a 2D run are:

```python
cfg.grid.nx = 256
cfg.grid.ny = 256
cfg.grid.ndim = 2
cfg.grid.lx = 100.0
cfg.grid.ly = 100.0
cfg.grid.grid_type = "periodic"
```

Supported `grid_type` values:

- `periodic`
- `closed-interval`

Use `periodic` for FFT-based solvers and clean spectral behavior. Use `closed-interval` when you need finite-domain behavior, but prefer FDM-family solvers for the best agreement on that grid type.

### One-dimensional runs

Set `cfg.grid.ndim = 1`, then choose `nx` and `lx`. The `ny` and `ly` settings are ignored. The resulting grid has `shape == (nx,)`, `spacing == (dx,)`, and no y-axis attributes. Public fields have shape `(nx,)` or `(B, nx)` for a batch; solvers and reservoirs use `(B, 1, nx)` internally.

```python
cfg.grid.ndim = 1
cfg.grid.nx = 512
cfg.grid.lx = 80.0
cfg.grid.grid_type = "periodic"
```

Every built-in solver supports 1D on its supported boundary type. The `isotropic-9pt` CUDA Laplacian is invalid in 1D. Live visualization and animation currently require 2D; saved HDF5 and NPY fields can be used to plot a 1D line. Potential and laser functions still receive `(X, Y)`; in 1D, `Y` is a zero array, so a 2D profile is sampled at y = 0.

## Physics constants

The `physics` block controls decay, interaction, and reservoir coupling parameters such as:

- `gamma_C`, `gamma_R`
- `g_C`, `g_R`
- `R`
- `m_eff`
- `hbar`
- `init_eps`
- `init_mode`, `init_k_cutoff_um`, `init_seed`
- `kinetic_relaxation_eta`
- `reservoir_diffusion_I`, `reservoir_diffusion_A`, `reservoir_diffusion_R`

These values directly enter the nonlinear GPE-reservoir dynamics. Keep units consistent across the full configuration; the package does not perform unit conversion for you.

`init_mode` controls the initial condensate seed.  The default
`legacy_positive_uniform` reproduces historical runs.  For geometry-sensitive
studies, `complex_gaussian_zero_mean` or `filtered_complex_gaussian` avoids the
biased positive-uniform seed; filtered mode requires `init_k_cutoff_um` and
records that cutoff in metadata.

`kinetic_relaxation_eta` enables a phenomenological kinetic-energy relaxation
term for the condensate.  It defaults to `0.0`.  When enabled it damps high-k
condensate modes proportionally to active reservoir density and `k^2`, so it
should be chosen from convergence and k-space diagnostics rather than from
visual appearance alone.

The reservoir diffusion coefficients also default to `0.0`.  They add physical
transport to reservoir fields (`nI`, `nA`, or `nR`, depending on reservoir
model), using periodic wrapping on periodic grids and mirrored Neumann behavior
on `closed-interval` grids.

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
- For `rk4-cuda`, `cfg.solver.laplacian` can be `five-point` or
  `isotropic-9pt`; the latter reduces grid-direction anisotropy on square cells.

## Backend selection

Request the GPU backend with:

```python
cfg.compute_engine.use_gpu = True
```

`gpu_device` selects the CUDA device by index among the visible devices; see
[Running on GPUs](running-on-gpus.md#device-selection). If CuPy or a CUDA device is unavailable, the backend
falls back to NumPy. Validate CUDA paths on the target hardware.

## YAML configuration

Save or load the complete configuration with `Config.to_yaml` and
`Config.from_yaml`:

```python
from polarism import Config

cfg = Config()
cfg.grid.nx = 128
cfg.to_yaml("run.yaml")
restored = Config.from_yaml("run.yaml")
```

The YAML file uses the same nested sections and field names as `Config`.
Unknown keys raise an error with the full path, such as `grid.typo`.
Loaded values pass through configuration validation.

Run the file from the CLI and override individual fields with flags:

```bash
polarism --config run.yaml --grid.nx 256
```

`--sweep sweep.yaml` can be used alongside `--config`.
