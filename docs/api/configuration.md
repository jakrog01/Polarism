# Configuration

`Config` is the root user-facing configuration object. It is composed of dataclass blocks so that physics parameters, numerics, output controls, and backend choices stay separated.

## Top-level blocks

| Block | Purpose |
| --- | --- |
| `grid` | spatial resolution, box size, and grid topology |
| `boundary_condition` | absorbing boundary settings |
| `potential` | external potential type and parameters |
| `physics` | driven-dissipative GPE and reservoir coefficients |
| `laser` | pump mode and pump-shape parameters |
| `reservoir` | reservoir model selection |
| `solver` | time integrator and timestep settings |
| `result` | storage, visualization, and output cadence |
| `compute_engine` | CPU/GPU backend selection |

## Grid options

| Field | Meaning | Main options |
| --- | --- | --- |
| `grid_type` | spatial topology | `periodic`, `closed-interval` |
| `nx`, `ny` | number of grid points | positive integers |
| `lx`, `ly` | physical box lengths | positive floats |

## Validation

`validate_config(cfg)` raises `ConfigValidationError` for physically invalid
values and issues `UserWarning` for numerically risky but not strictly invalid
configurations.  Checks include:

- `hbar > 0`, `m_eff > 0`
- all decay/scattering/transfer rates (`gamma_C`, `gamma_R`, `gamma_I`,
  `gamma_A`, `R`, `R_IA`, `R_AI`, `kappa`) are `>= 0`
- all interaction, relaxation, and diffusion coefficients (`g_C`, `g_R`,
  `g_I`, `kinetic_relaxation_eta`, `reservoir_diffusion_I/A/R`) are `>= 0`
- `init_eps >= 0`; `init_mode` is a recognised value; `init_k_cutoff_um > 0`
  is required when `init_mode='filtered_complex_gaussian'`
- `boundary_condition.strength >= 0`; `mask_width_percent` in `[0, 0.5]`;
  `absorption` must be a registered strategy; `profile_type` is in
  `{'sin2', 'parabolic'}` when absorption is active
- `laser.sigma_space > 0`; `sigma_time > 0` and `cutoff_sigma > 0` for
  `pulse-gaussian`; `pulse_separation > 0` when `n_pulses > 1`; `n_pulses >= 0`
- `power_definition` in `{'peak_amplitude', 'pulse_energy'}`

Stability warnings are emitted for explicit FDM/RK4 solvers when:

- `solver.dt` exceeds the RK4 kinetic-term threshold
  `~sqrt(2) * m_eff * dx_min^2 / (2 * hbar)`
- `solver.dt` exceeds the reservoir-diffusion stability limit
  `~0.348 * dx_min^2 / D_max` for any active `reservoir_diffusion_*` coefficient
- `solver.dt` exceeds the effective diffusion threshold for
  `kinetic_relaxation_eta`, computed as
  `~0.348 * dx_min^2 / (eta * n_active_ref * hbar / 2m)`, with
  `n_active_ref = max(1, gamma_C/R)` or `1` when `R=0`

Warnings include the concrete values of `dt`, `dx_min`, and the offending
coefficient so that the corrective action is unambiguous.

## Boundary-condition options

| Field | Meaning | Main options |
| --- | --- | --- |
| `absorption` | absorbing-boundary strategy | `no-absorption`, `mask`, `cap` |
| `profile_type` | absorption profile shape | `sin2`, `parabolic` |
| `strength` | boundary damping strength; `0` disables damping. For `mask`, the per-step multiplier is `(1 - profile) ** strength`; for `cap`, it scales the imaginary absorbing potential. | nonnegative float |
| `mask_width_percent` | boundary-layer width as fraction of domain | float in `[0, 0.5]` |

## Potential options

| Field | Meaning | Main options |
| --- | --- | --- |
| `potential_type` | external potential model | `zero`, `double-well-supergaussian` |
| `x1`, `y1`, `x2`, `y2` | feature positions for structured potentials | floats |
| `V1`, `V2` | potential amplitudes | floats |
| `w1`, `w2` | characteristic widths | positive floats |
| `order` | super-Gaussian order | positive float |
| `expose_results` | retained output-exposure setting | currently has no effect on result-node creation |

## Laser options

| Field | Meaning | Main options |
| --- | --- | --- |
| `mode` | single or multi-pump setup | `single`, `multiple` |
| `config_file` | YAML file for multi-pump setups | path string |
| `laser_type` | pump-profile selection | `uniform`, `continuous-gaussian`, `continuous-exp`, `continuous-exp-length`, `pulse-gaussian` |
| `P0`, `Pmax` | pump strength parameters; for `pulse-gaussian`, their physical meaning is set by `power_definition` | floats |
| `power_definition` | interpretation of `P0`, `Pmax`, and per-laser `power` for `pulse-gaussian` | `peak_amplitude`, `pulse_energy` |
| `x0`, `y0` | pump center | floats |
| `sigma_space` | spatial width scale | positive float |
| `sigma_time` | pulse width in time | positive float |
| `pulse_separation` | distance between pulses | positive float |
| `cutoff_sigma` | pulse truncation radius in sigma units | positive float |
| `delay` | start delay before pump turns on; for `pulse-gaussian`, the first peak occurs later at `delay + cutoff_sigma * sigma_time` | nonnegative float |
| `n_pulses` | finite pulse-train length for pulsed lasers; `0` means an unbounded pulse train | nonnegative integer |
| `expose_results` | publish the total pump field and per-laser scalar group to result visitors | boolean |

`power_definition` is currently used by `pulse-gaussian`.  In
`peak_amplitude` mode, `P0` is the local peak source density at the pump
centre, so changing `sigma_space` at fixed `P0` changes the total injected
dose.  In `pulse_energy` mode, `P0` is the integrated per-pulse dose: the
spatial Gaussian is normalized by its discrete grid integral and the temporal
Gaussian by its truncated analytic integral.  Use `pulse_energy` for spot-size
or geometry sweeps where different `sigma_space` values must receive the same
total pulse dose.

## Reservoir options

| Field | Meaning | Main options |
| --- | --- | --- |
| `reservoir_type` | reservoir-model selection | `single`, `double`, `quadratic-double` |
| `expose_results` | publish reservoir fields and scalars to result visitors | boolean |

`quadratic-double` uses an inactive reservoir directly fed by the pump and an
active reservoir that feeds the condensate:

```text
dnI/dt = P(x,y,t) - kappa*nI^2 - gamma_I*nI
dnR/dt = kappa*nI^2 - gamma_R*nR - R*nR*|psi|^2
```

The condensate equation receives `nR` as the active reservoir density.

## Solver options

| Field | Meaning | Main options |
| --- | --- | --- |
| `method` | time integrator | `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda`, `split-step-fft`, `etd-rk2`, `ip-rk4`, `ifrk4-fft-cuda` |
| `dt` | timestep | positive float |
| `total_time` | total simulation time | positive float |
| `precision` | arithmetic mode where supported | typically `single`, `double` |
| `laplacian` | finite-difference Laplacian stencil for `rk4-cuda` | `five-point`, `isotropic-9pt` |

`solver.laplacian` defaults to `five-point`.  `isotropic-9pt` is available for
square cells (`dx == dy`) and discretizes the same physical operator, `nabla^2`,
with reduced grid-direction anisotropy.

## Result options

| Field | Meaning | Main options |
| --- | --- | --- |
| `real_time_view` | live visualization during run | boolean |
| `real_time_refresh_interval` | UI refresh cadence | positive float |
| `save_results` | enable result processing | boolean |
| `save_hdf5`, `save_json`, `save_npy` | storage backends | booleans |
| `save_interval` | number of steps between writes | positive integer |
| `batch_size` | buffered result chunk size | positive integer |
| `output_directory` | output path | directory string |
| `animate` | stream an online animation during the run | boolean |
| `animation_fps` | encoded movie frame rate | positive integer |
| `animation_target_seconds` | nominal target duration used for logging | positive integer |
| `animation_fields` | optional tuple of result-node names to include | tuple of strings |
| `animation_backend` | video backend selector | `auto`, `ffmpeg`; `pynvvideocodec` currently fails explicitly |
| `animation_encoder` | preferred ffmpeg encoder | e.g. `h264_nvenc`, `libx264`, `ffv1`, `mpeg4` |
| `animation_output` | output movie path | path string or `None` |

`animate` uses the visitor-based results path.  The animation visitor receives
device arrays during GPU runs and is marked non-fatal: encoder failures disable
only the movie stream, while storage visitors remain fatal by default.

## Compute-engine options

| Field | Meaning | Main options |
| --- | --- | --- |
| `use_gpu` | request CuPy/CUDA backend | boolean |
| `gpu_device` | CUDA device index | nonnegative integer |

## Physics block

The `physics` block stores the scalar coefficients entering the condensate and reservoir equations, including:

- `hbar`, `m_eff`
- `gamma_C`, `gamma_R`, `gamma_I`, `gamma_A`
- `g_C`, `g_R`, `g_I`
- `R`, `R_IA`, `R_AI`
- `kappa` for the `quadratic-double` reservoir
- `init_eps`, `init_mode`, `init_k_cutoff_um`, `init_seed`
- `kinetic_relaxation_eta`
- `reservoir_diffusion_I`, `reservoir_diffusion_A`, `reservoir_diffusion_R`

These values are model parameters, not high-level feature flags, so they should be changed with physical units and stability constraints in mind.

The default `src/polariton_hpc_pipeline/config.yaml` is a GaAs/AlGaAs
microcavity preset for a pulsed `quadratic-double` reservoir campaign.  Its
`g_C`, `g_R`, and `R` values are scaled from a GaAs exciton-interaction estimate
with `N_QW = 4` and exciton fraction `|X|^2 = 0.40`; they should not be treated
as generic defaults for other materials.

Initial-condition options:

| Field | Meaning | Main options |
| --- | --- | --- |
| `init_eps` | amplitude scale of the initial condensate seed | positive float |
| `init_mode` | initial seed generator | `legacy_positive_uniform`, `complex_gaussian_zero_mean`, `filtered_complex_gaussian` |
| `init_k_cutoff_um` | radial cutoff in rad/um for `filtered_complex_gaussian` | positive float, required for filtered mode |
| `init_seed` | optional RNG seed | integer or `null` |
| `kinetic_relaxation_eta` | condensate kinetic-energy relaxation strength | nonnegative float, default `0.0` |
| `reservoir_diffusion_I` | inactive-reservoir diffusion coefficient | nonnegative float, default `0.0` |
| `reservoir_diffusion_A` | active-density diffusion coefficient for `double` | nonnegative float, default `0.0` |
| `reservoir_diffusion_R` | active-density diffusion coefficient for `single` and `quadratic-double` | nonnegative float, default `0.0` |

The legacy mode is kept for reproducibility.  For geometry-sensitive runs, prefer
zero-mean or filtered seeds and record the cutoff in run metadata.

Relaxation and diffusion are model terms, not visualization filters.  They are
disabled by default and should be enabled only with a stated physical rationale
and checked against the recorded k-space diagnostics.
