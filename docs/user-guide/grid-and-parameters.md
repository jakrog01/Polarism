# Grid and Parameters

The full runtime configuration is held in `polarism.config.simulation_parameters.Config`, a nested dataclass tree that separates numerical, physical, and output concerns.

## Main configuration sections

| Section | Purpose |
| --- | --- |
| `grid` | Spatial resolution, box size, and grid topology |
| `physics` | Material constants, interaction strengths, and decay rates |
| `terms` | Optional extra terms added to the condensate equation |
| `noise` | Initial field, seed, and stochastic Wigner increments |
| `initial_condition` | Deterministic start field added to the initial noise |
| `boundary_condition` | Absorption type, profile, and mask width |
| `potential` | External potential family and parameters |
| `laser` | Pump definition for single or multiple sources |
| `reservoir` | Single, double, quadratic-double, or no reservoir |
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
- `kinetic_relaxation_eta`
- `reservoir_diffusion_I`, `reservoir_diffusion_A`, `reservoir_diffusion_R`
- `dispersion`, the name of the kinetic-energy relation (`parabolic` by default; see [Dispersion plugins](../plugins/dispersion.md))

These values directly enter the nonlinear GPE-reservoir dynamics. Keep units consistent across the full configuration; the package does not perform unit conversion for you.

## Noise

The `noise` block controls the initial condensate field and optional noise
during evolution:

```yaml
noise:
  enabled: true
  seed: null
  initial: complex_gaussian_zero_mean
  initial_amplitude: 1.0e-3
  initial_k_cutoff_um: null
  stochastic: none
  stochastic_strength: 1.0
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Set to `false` for an exactly zero initial field and no stochastic increments |
| `seed` | `null` | Non-negative int64 seed; `null` resolves from OS entropy and is recorded |
| `initial` | `complex_gaussian_zero_mean` | `none`, `legacy_positive_uniform`, `complex_gaussian_zero_mean`, or `filtered_complex_gaussian` |
| `initial_amplitude` | `1.0e-3` | Amplitude of the initial field |
| `initial_k_cutoff_um` | `null` | Required positive radial cutoff for filtered initial fields |
| `stochastic` | `none` | `none` or `wigner` |
| `stochastic_strength` | `1.0` | Non-negative multiplier of Wigner noise variance; zero disables increments |

The default `complex_gaussian_zero_mean` is unbiased white noise with RMS
amplitude `initial_amplitude`. `legacy_positive_uniform`, the default before
0.7, has a non-zero mean, so it also seeds a uniform k = 0 component; use it
only to reproduce old runs. For a zero field
with stochastic growth, set `initial: none` and `stochastic: wigner`. For a
fully deterministic zero-field run, set `enabled: false`. The initial field
uses NumPy's seeded generator on both devices.

One seed controls the run. With `seed: null` a seed is drawn once per run (or
once per sweep) and recorded in the run's config, results metadata and
checkpoint, so any run can be repeated by setting that seed. All members of a
parameter sweep share the seed, so they see identical random numbers and differ
only by the swept parameters. Sweep `noise.seed` over values to create a noise
ensemble. A sweep member reproduces bit-for-bit as a single run with the same
config, whatever the batch size, and after a checkpoint resume.
For a single run, `--noise.seed 42` sets the seed from the command line.

With `stochastic: wigner`, a cell receives
$\sqrt{s(\gamma_C+R n_A)dt/(4\Delta V)}(z_{re}+i z_{im})$ after each step,
where $z_{re}$ and $z_{im}$ are independent unit normals, $s$ is
`stochastic_strength`, and $\Delta V$ is `dx` in 1D or `dx dy` in 2D.
This implementation omits the Wigner energy and reservoir corrections; see
[Physics Background](../development/physics-background.md#initial-noise-and-stochastic-seeds).

`kinetic_relaxation_eta` enables a phenomenological kinetic-energy relaxation
term for the condensate.  It defaults to `0.0`.  When enabled it damps high-k
condensate modes proportionally to active reservoir density and `k^2`, so it
should be chosen from convergence and k-space diagnostics rather than from
visual appearance alone.

The reservoir diffusion coefficients also default to `0.0`.  They add physical
transport to reservoir fields (`nI`, `nA`, or `nR`, depending on reservoir
model), using periodic wrapping on periodic grids and mirrored Neumann behavior
on `closed-interval` grids.

## Initial conditions

The start field is the deterministic `initial_condition` plus the random field
from `noise.initial` (when `noise.enabled`). The default `type: zero` adds
nothing, so the start field is exactly the noise field and existing configs
are unchanged. With `noise.enabled: false` the start field is the
deterministic part alone.

```yaml
initial_condition:
  type: vortex
  amplitude: 1.0
  x0: 0.0
  y0: 0.0
  charge: 1
  core_width: 1.0
```

| Field | Default | Meaning |
| --- | --- | --- |
| `type` | `zero` | `zero`, `plane-wave`, `vortex`, `file`, or a registered [plugin](../plugins/initial-conditions.md) name |
| `amplitude` | `1.0` | Field amplitude, √density (µm⁻¹ in 2D); non-negative |
| `x0`, `y0` | `0.0` | Centre of the vortex in µm; the grid is centred on the origin |
| `kx`, `ky` | `0.0` | Wave vector of the plane wave in rad/µm; 1D uses `kx` only |
| `charge` | `1` | Integer winding number of the vortex; may be negative |
| `core_width` | `1.0` | Vortex healing length in µm; positive |
| `path` | `null` | Source file for `type: file` |
| `frame` | `-1` | Saved frame read from a results file; `-1` is the last |
| `extra` | `{}` | Free-form keys for plugin conditions |

| `type` | Start field |
| --- | --- |
| `zero` | $0$ |
| `plane-wave` | $A\,e^{i(k_x x + k_y y)}$ |
| `vortex` | $A \tanh(r/w)\,e^{i m \theta}$ with $r,\theta$ measured from $(x_0, y_0)$; 2D only |
| `file` | The field stored in `path`; `amplitude` is not applied |

`type: file` reads a `.npy` array of the grid's shape, a Polarism `results.h5`
that saved `psi` (`result.save_psi`, member 0 when it holds a batch, frame
`frame`), or a checkpoint (member 0). Its shape must equal the grid's exactly;
there is no interpolation, and a mismatch is an error.

In a parameter sweep, `initial_condition.amplitude`, `x0`, `y0`, `kx`, `ky`,
`charge`, `core_width`, and `frame` can vary inside a batch, while
`initial_condition.type` and `initial_condition.path` form separate groups.
Each member adds its own deterministic part to its own noise field.

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
