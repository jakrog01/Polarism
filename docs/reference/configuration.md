# Configuration reference

Use this page to check the exact fields, defaults and constraints of `ps.Config()`. Values use ps for time, µm for distance, meV for energy, and the units shown in each table.

Python assignment rejects an unknown public dataclass field with `AttributeError` and a close-name suggestion. `Config.from_yaml()` rejects unknown section and field names with `ValueError`; every section must be a mapping. `terms` is a mapping of registered term names to parameter mappings. Only `grid`, `initial_condition`, `laser` and `reservoir` provide an `extra` mapping for plugin-specific keys. YAML lists assigned to tuple fields become tuples. Call `polarism.config.validation.validate_config(cfg)` to validate a Python config before constructing the controller; `Config.from_yaml()` validates automatically.

## `Config`

| Block | Meaning |
| --- | --- |
| [`grid`](#grid) | Dimension, box size and grid topology |
| [`physics`](#physics) | Mass, interactions, rates, dispersion and spin |
| [`laser`](#laser) | Incoherent pump |
| [`reservoir`](#reservoir) | Reservoir model and component count |
| [`potential`](#potential) | External potential and its time envelope |
| [`drive`](#drive) | Coherent source and rotating frame |
| [`initial_condition`](#initial_condition), [`noise`](#noise) | Starting field and random increments |
| [`boundary_condition`](#boundary_condition) | Edge absorption |
| [`solver`](#solver), [`compute_engine`](#compute_engine) | Time integration and CPU or GPU |
| [`result`](#result) | Fields, observables, visualization and storage |
| [`checkpoint`](#checkpoint), [`stop`](#stop), [`sweep`](#sweep) | Restart, early stopping and sweep controls |
| [`terms`](#terms) | Additional registered model terms |

## `grid`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `nx` | `int` | `256` | 1 | integer ≥ 1 | Number of x samples. |
| `ny` | `int` | `256` | 1 | integer ≥ 1 in 2D | Number of y samples; unused in 1D. |
| `lx` | `float` | `100.0` | µm | > 0 | Box length along x. |
| `ly` | `float` | `100.0` | µm | > 0 in 2D | Box length along y; unused in 1D. |
| `grid_type` | `str` | `periodic` | n/a | registered; built in: periodic, closed-interval | Grid topology or plugin name. |
| `ndim` | `int` | `2` | 1 | 1 or 2 | Number of spatial dimensions. |
| `extra` | `dict[str, Any]` | `{}` | n/a | mapping | Plugin grid parameters. |

## `boundary_condition`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `profile_type` | `str` | `sin2` | n/a | sin2, parabolic when active | Absorbing layer profile. |
| `strength` | `float` | `1.0` | ps⁻¹ | ≥ 0 | Density damping rate at the edge. |
| `absorption` | `str` | `no-absorption` | n/a | registered; built in: no-absorption, mask, cap | Absorption strategy. |
| `mask_width_percent` | `float` | `0.2` | fraction | 0 to 0.5 | Width of each absorbing edge as a fraction of box length. |

## `potential`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `expose_results` | `bool` | true | 1 | boolean | Legacy output setting; potential result exposure is currently unaffected. |
| `potential_type` | `str` | `zero` | n/a | registered; built in: zero, double-well-supergaussian, profile | Spatial potential factory. |
| `profile` | `dict[str, Any]` | `{}` | n/a | profile mapping; built in: uniform, gaussian, supergaussian, ring, lattice, file; required for profile type | Spatial energy profile specification. |
| `envelope` | `dict[str, Any]` | `{type: constant}` | n/a | registered envelope mapping; built in: constant, step, gaussian-pulses, data | Time envelope specification. |
| `x1` | `float` | `-2.0` | µm | finite number | First well x position. |
| `y1` | `float` | `0.0` | µm | finite number | First well y position. |
| `x2` | `float` | `2.0` | µm | finite number | Second well x position. |
| `y2` | `float` | `0.0` | µm | finite number | Second well y position. |
| `V1` | `float` | `-0.002` | meV | number | First well energy scale. |
| `V2` | `float` | `-0.0022` | meV | number | Second well energy scale. |
| `w1` | `float` | `1.5` | µm | number | First well width. |
| `w2` | `float` | `1.5` | µm | number | Second well width. |
| `order` | `float` | `2.0` | 1 | number | Super-Gaussian exponent. |

## `drive`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `bool` | false | 1 | boolean | Enable coherent drive. |
| `amplitude` | `float` | `0.0` | meV·µm⁻¹ | finite number | Coherent source amplitude in 2D; scales with field units in 1D. |
| `energy` | `float` | `0.0` | meV | finite number | Drive energy in rotating frame. |
| `kx` | `float` | `0.0` | rad/µm | finite number | Drive x wave vector. |
| `ky` | `float` | `0.0` | rad/µm | finite number; zero in 1D | Drive y wave vector. |
| `polarization` | `list[complex]` | `[1, 0]` | 1 | two finite complex values with nonzero norm | Jones amplitudes for spin components. |
| `profile` | `dict[str, Any]` | `{type: uniform}` | n/a | valid spatial profile mapping; built in: uniform, gaussian, supergaussian, ring, lattice, file | Drive spatial profile. |
| `envelope` | `dict[str, Any]` | `{type: constant}` | n/a | valid temporal envelope mapping; built in: constant, step, gaussian-pulses, data | Drive time envelope. |

## `physics`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `hbar` | `float` | `0.6582119514` | meV·ps | > 0 | Reduced Planck constant. |
| `m_eff` | `float` | `0.284` | meV·ps²/µm² | > 0 | Effective mass in simulation units. |
| `gamma_R` | `float` | `0.005` | ps⁻¹ | ≥ 0 | Decay rate of active n_R in single and quadratic-double reservoirs. |
| `gamma_C` | `float` | `0.083` | ps⁻¹ | ≥ 0 | Condensate decay rate. |
| `g_C` | `float` | `0.001` | meV·µm² | ≥ 0 | Same-spin condensate interaction in 2D. |
| `g_R` | `float` | `0.002` | meV·µm² | ≥ 0 | Active reservoir energy shift in 2D. |
| `g_I` | `float` | `0.0` | meV·µm² | ≥ 0 | Inactive reservoir energy shift in 2D. |
| `gamma_I` | `float` | `0.001` | ps⁻¹ | ≥ 0 | Decay rate of inactive n_I. |
| `gamma_A` | `float` | `0.005` | ps⁻¹ | ≥ 0 | Decay rate of active n_A in the double reservoir. |
| `R` | `float` | `0.02` | µm²/ps | ≥ 0 | Reservoir stimulated scattering in 2D. |
| `R_IA` | `float` | `0.05` | ps⁻¹ | ≥ 0 | Inactive-to-active transfer rate. |
| `R_AI` | `float` | `0.0025` | ps⁻¹ | ≥ 0 | Active-to-inactive transfer rate. |
| `kappa` | `float` | `0.05` | µm²/ps | ≥ 0 | Quadratic inactive-to-active transfer coefficient in 2D. |
| `kinetic_relaxation_eta` | `float` | `0.0` | µm² in 2D; µm in 1D | ≥ 0 | Reservoir-dependent kinetic relaxation coefficient. |
| `reservoir_diffusion_I` | `float` | `0.0` | µm²/ps | ≥ 0 | Inactive reservoir diffusion coefficient. |
| `reservoir_diffusion_A` | `float` | `0.0` | µm²/ps | ≥ 0 | Active density diffusion coefficient for double reservoir. |
| `reservoir_diffusion_R` | `float` | `0.0` | µm²/ps | ≥ 0 | Single or quadratic-double reservoir diffusion coefficient. |
| `dispersion` | `str` | `parabolic` | n/a | registered; built in: parabolic | Kinetic energy relation. |
| `spin` | `str` | `scalar` | n/a | scalar, spinor | One or two condensate components. |
| `alpha_2` | `float` | `0.0` | meV·µm² | finite number | Cross-spin condensate interaction in 2D. |
| `zeeman` | `float` | `0.0` | meV | finite number | Spin energy splitting. |
| `te_tm` | `float` | `0.0` | meV·µm² | finite number | TE-TM coupling coefficient. |

## `terms`

Type: `dict[str, Any]`. Default: `{}`. Units and allowed keys depend on registered [term plugins](../plugins/terms.md). Each value must be a parameter mapping. The key is a registered term name.

## `noise`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `enabled` | `bool` | true | 1 | boolean | Enable initial and stochastic noise. |
| `seed` | `int | None` | null | 1 | null or integer 0 to 2⁶³−1 | Random seed; null draws entropy. |
| `initial` | `str` | `complex_gaussian_zero_mean` | n/a | none, legacy_positive_uniform, complex_gaussian_zero_mean, filtered_complex_gaussian | Initial random field distribution. |
| `initial_amplitude` | `float` | `0.001` | µm⁻¹ | finite ≥ 0 in 2D | Initial field amplitude; field units change in 1D. |
| `initial_k_cutoff_um` | `float | None` | null | rad/µm | null or finite > 0 for filtered mode | Initial noise wave number cutoff. |
| `stochastic` | `str` | `none` | n/a | none, wigner | Per-step stochastic noise model. |
| `stochastic_strength` | `float` | `1.0` | 1 | finite ≥ 0 | Multiplier of Wigner variance. |

## `initial_condition`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `type` | `str` | `zero` | n/a | registered; built in: zero, plane-wave, vortex, file | Deterministic initial field. |
| `amplitude` | `float` | `1.0` | µm⁻¹ | finite ≥ 0 in 2D | Field amplitude, unused for file input. |
| `x0` | `float` | `0.0` | µm | finite number | Vortex x center. |
| `y0` | `float` | `0.0` | µm | finite number | Vortex y center. |
| `kx` | `float` | `0.0` | rad/µm | finite number | Plane-wave x wave vector. |
| `ky` | `float` | `0.0` | rad/µm | finite number | Plane-wave y wave vector. |
| `charge` | `int` | `1` | 1 | integer | Vortex winding number. |
| `core_width` | `float` | `1.0` | µm | finite > 0 | Vortex core width. |
| `path` | `str | None` | null | path | null or nonempty string; required for file | Input NPY, results HDF5 or checkpoint path. |
| `frame` | `int` | `-1` | 1 | integer | Results frame index for file input. |
| `extra` | `dict[str, Any]` | `{}` | n/a | mapping | Plugin initial condition parameters. |

## `laser`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `expose_results` | `bool` | true | 1 | boolean | Expose pump result nodes. |
| `mode` | `str` | `single` | n/a | single, multiple | One laser or YAML list of lasers. |
| `config_file` | `str | None` | null | path | null or path string | Multiple-laser YAML path. |
| `laser_type` | `str` | `pulse-gaussian` | n/a | registered; built in: uniform, continuous-gaussian, continuous-exp, continuous-exp-length, pulse-gaussian, profile | Laser factory name. |
| `profile` | `dict[str, Any]` | `{}` | n/a | profile mapping; built in: uniform, gaussian, supergaussian, ring, lattice, file; required for profile laser | Spatial pump profile. |
| `envelope` | `dict[str, Any]` | `{type: constant}` | n/a | registered envelope mapping; built in: constant, step, gaussian-pulses, data | Pump time envelope. |
| `P0` | `float` | `0.4` | µm⁻²·ps⁻¹ or particles/pulse | number | Base pump source density in 2D; pulse-energy mode changes units. |
| `circular_polarization` | `float` | `0.0` | 1 | finite from −1 to 1 | Relative spin pump polarization. |
| `Pmax` | `float` | `1.2` | µm⁻²·ps⁻¹ or particles/pulse | number | Maximum pump source density in 2D; pulse-energy mode changes units. |
| `x0` | `float` | `0.0` | µm | number | Pump x center. |
| `y0` | `float` | `0.0` | µm | number | Pump y center. |
| `sigma_space` | `float` | `15.0` | µm | > 0 | Spatial width. |
| `sigma_time` | `float` | `0.1` | ps | > 0 for pulse-gaussian | Pulse time width. |
| `pulse_separation` | `float` | `0.6` | ps | > 0 when n_pulses > 1 | Separation between pulses. |
| `cutoff_sigma` | `float` | `3.0` | 1 | > 0 for pulse-gaussian | Pulse truncation radius in time widths. |
| `delay` | `float` | `0.0` | ps | number | Time offset before pump starts. |
| `n_pulses` | `int` | `0` | 1 | integer ≥ 0 for pulse-gaussian | Number of pulses; zero means an unlimited train. |
| `power_definition` | `str` | `peak_amplitude` | n/a | peak_amplitude, pulse_energy | Interpretation of pulsed pump strength. |
| `extra` | `dict[str, Any]` | `{}` | n/a | mapping | Plugin laser parameters. |

## `reservoir`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `expose_results` | `bool` | true | 1 | boolean | Expose reservoir result nodes. |
| `reservoir_type` | `str` | `double` | n/a | registered; built in: single, double, quadratic-double, none | Reservoir model. |
| `components` | `int | None` | null | 1 | null, 1, 2; cannot exceed condensate components | Number of reservoir spin channels. |
| `extra` | `dict[str, Any]` | `{}` | n/a | mapping | Plugin reservoir parameters. |

## `solver`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `total_time` | `float` | `5.0` | ps | > 0 | End time measured from initial state. |
| `dt` | `float` | `0.001` | ps | > 0 | Schedule tick and fixed step size. |
| `method` | `str` | `rk4-fdm` | n/a | registered solver name | Time integrator. |
| `tableau` | `str` | `rk4` | n/a | rk4, dp5, dop853, vern7, vern9 | Runge-Kutta coefficients for ifrk-fft-cuda. |
| `precision` | `str` | `double` | n/a | single, double | Floating point precision. |
| `laplacian` | `str` | `five-point` | n/a | five-point, isotropic-9pt for rk4-cuda | Finite difference stencil choice. |
| `cuda_graphs` | `bool` | true | 1 | boolean | Use CUDA graphs where eligible. |
| `adaptive` | `bool` | false | 1 | boolean; ifrk-fft-cuda with embedded tableau | Enable adaptive substeps. |
| `rtol` | `float` | `1e-06` | 1 | finite > 0; ≥ 1e-5 in single precision | Adaptive relative error target. |
| `atol` | `float` | `1e-08` | field units | finite ≥ 0 | Adaptive absolute error target. |
| `dt_max` | `float | None` | null | ps | null or finite > 0 | Maximum adaptive substep. |

## `result`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `real_time_view` | `bool` | false | 1 | boolean; 2D only | Display fields while running. |
| `view_member` | `int` | `0` | 1 | nonnegative integer | Batch member shown in live view. |
| `real_time_refresh_interval` | `float` | `0.1` | ps | number | Time between live view refreshes. |
| `save_results` | `bool` | false | 1 | boolean | Enable result processing and storage. |
| `save_hdf5` | `bool` | false | 1 | boolean | Legacy alias of `formats: [hdf5]`. |
| `save_json` | `bool` | false | 1 | boolean | Legacy alias of `formats: [json]`. |
| `save_npy` | `bool` | false | 1 | boolean | Legacy alias of `formats: [npy]`. |
| `formats` | `tuple[str, ...]` | `()` | n/a | tuple/list of registered names; built in: hdf5, npy, json | Selected storage formats. |
| `observables` | `tuple[str, ...]` | `()` | n/a | tuple/list of registered names; built in: angular-momentum, center-of-mass, current, momentum-density, vortices | Extra frame observables. |
| `accumulators` | `dict[str, dict]` | `{}` | n/a | mapping of registered names; built in: momentum-distribution, g2, spectrum-k, spectrum-x | Final sampled results such as spectra. |
| `compression` | `str` | `none` | n/a | none, lzf, gzip | Frame compression method. |
| `compression_level` | `int` | `4` | 1 | integer 0 to 9 | Gzip level. |
| `save_psi` | `bool` | false | 1 | boolean | Save complex condensate field. |
| `save_every` | `float | None` | null | ps | null or positive multiple of solver.dt | Frame sampling interval. |
| `batch_size` | `int` | `1000` | frames | integer ≥ 1 | Frames buffered before writing. |
| `output_directory` | `str` | `simulation_results` | path | string | Parent directory of run output. |
| `animate` | `bool` | false | 1 | boolean; 2D only | Encode animation during run. |
| `animation_fps` | `int` | `8` | frames/s | integer | Animation frame rate. |
| `animation_target_seconds` | `int` | `60` | s | integer | Nominal animation duration. |
| `animation_fields` | `tuple[str, ...]` | `()` | n/a | tuple of result names | Fields included in animation. |
| `animation_backend` | `str` | `auto` | n/a | auto, ffmpeg; pynvvideocodec errors | Encoder backend. |
| `animation_encoder` | `str` | `h264_nvenc` | n/a | ffmpeg encoder name | Preferred video codec. |
| `animation_output` | `str | None` | null | path | null or path string | Movie destination. |

## `compute_engine`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `use_gpu` | `bool` | false | 1 | boolean | Request CuPy backend. |
| `gpu_device` | `int` | `0` | 1 | integer device index | CUDA device number. |

## `checkpoint`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `every` | `float` | `0.0` | ps | zero or positive multiple of solver.dt | Periodic checkpoint interval; zero disables periodic saving. |
| `path` | `str | None` | null | path | null or nonempty string | Checkpoint destination. |
| `resume_from` | `str | None` | null | path | null or nonempty string | Checkpoint to restore. |
| `on_signal` | `bool` | true | 1 | boolean | Save and stop on SIGTERM or SIGUSR1. |

## `stop`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `steady_state` | `bool` | false | 1 | boolean | Stop after two steady comparisons. |
| `rtol` | `float` | `1e-06` | 1 | finite > 0 | Relative convergence threshold. |
| `check_every` | `float` | `10.0` | ps | positive multiple of solver.dt when enabled | Interval between checks. |
| `min_time` | `float` | `0.0` | ps | finite ≥ 0 | Earliest time eligible for a steady stop. |

## `sweep`

| Field | Type | Default | Units | Allowed values | Meaning |
| --- | --- | --- | --- | --- | --- |
| `batch_size` | `int | None` | null | members | null or positive integer | Members per sweep batch. |
| `memory_fraction` | `float` | `0.8` | fraction | > 0 and ≤ 1 | Fraction of available GPU memory allocated to a batch. |
| `max_batch` | `int | None` | null | members | null or positive integer | Maximum members per batch. |

## Choice guide

These choices keep the same field names as the tables above. For equations and detailed setup, follow the linked user guide pages.

| Field and value | Choose it when |
| --- | --- |
| `grid.grid_type: periodic` | The domain wraps, or an FFT solver is required. |
| `grid.grid_type: closed-interval` | The simulation needs explicit finite box edges and an FDM solver. |
| `boundary_condition.absorption: no-absorption` | Boundary damping is unnecessary. |
| `boundary_condition.absorption: mask` | Damping should be applied after each step at the edge. |
| `boundary_condition.absorption: cap` | Damping should enter as a complex potential. |
| `potential.potential_type: zero` | No external trap is needed. |
| `potential.potential_type: double-well-supergaussian` | Two localized wells with tunable widths and wall order are needed. |
| `potential.potential_type: profile` | A registered spatial profile defines the potential. |
| `laser.laser_type: uniform` | Pump strength is homogeneous. |
| `laser.laser_type: continuous-gaussian` | A continuous localized Gaussian spot is needed. |
| `laser.laser_type: continuous-exp` | A legacy exponential profile with long tails is needed. |
| `laser.laser_type: continuous-exp-length` | `sigma_space` should be the 1/e radial length in µm. |
| `laser.laser_type: pulse-gaussian` | A Gaussian pulse train is needed. |
| `laser.laser_type: profile` | A registered profile and envelope define the pump. |
| `laser.power_definition: peak_amplitude` | `P0` fixes local peak source density, in µm⁻² ps⁻¹ for 2D. |
| `laser.power_definition: pulse_energy` | `P0` fixes integrated particles per pulse for spot-size comparisons. |
| `laser.mode: single` | One pump definition is enough. |
| `laser.mode: multiple` | Several pumps are listed in an external YAML file. |
| `reservoir.reservoir_type: single` | One active reservoir density is sufficient. |
| `reservoir.reservoir_type: double` | Inactive and active populations need separate rates. |
| `reservoir.reservoir_type: quadratic-double` | Inactive-to-active feeding depends on the square of inactive density. |
| `reservoir.reservoir_type: none` | No incoherent reservoir is present. |
| `noise.initial: none` | The deterministic initial condition should be used alone. |
| `noise.initial: legacy_positive_uniform` | An old run needs reproduction. |
| `noise.initial: complex_gaussian_zero_mean` | An unbiased broadband random seed is wanted. |
| `noise.initial: filtered_complex_gaussian` | The initial seed needs a radial cutoff in rad/µm. |
| `compute_engine.use_gpu: false` | Run with NumPy on a CPU. |
| `compute_engine.use_gpu: true` | Run with CuPy on an available CUDA GPU. |

For methods, adaptive support and cost, use [Solver reference](solvers.md). For pump, reservoir and profile details, use [Pumps and reservoirs](../user-guide/pumps-and-reservoirs.md) and [Profiles and envelopes](../user-guide/profiles-and-envelopes.md).

## Validation details

`validate_config` raises `ConfigValidationError` for invalid values. Registry names are checked for dispersion, terms, initial conditions, storage formats, observables, accumulators and absorption. Grid topology and solver features are checked against the selected solver. Some plugin factories validate their own specifications during construction.

- A filtered initial noise field requires a positive `initial_k_cutoff_um` in rad/µm. `vortex` initial conditions need a 2D grid, and `file` needs a path.
- `drive.polarization` needs two finite complex amplitudes with nonzero norm. Complex YAML values should be quoted. Spin energy fields used with `spin: scalar` emit warnings when nonzero.
- `laser_type: profile` needs a nonempty profile and `peak_amplitude` power definition. In `pulse_energy` mode the pulse strength describes integrated dose, so its units differ from peak source density.
- `result.save_every`, checkpoint intervals and accumulator sampling times must be exact positive multiples of `solver.dt` in ps. Accumulator `start` and `stop` may be zero. `stop` cannot precede `start`. Spectra require an even `segment` of at least 8 and enough samples. `spectrum-x.y` is in µm and requires 2D.
- Adaptive stepping requires `ifrk-fft-cuda`, an embedded tableau (`dp5`, `dop853`, `vern7` or `vern9`), stochastic noise off and an absorption strategy whose `after_step` action is a no-op. `dt` remains the output and checkpoint schedule tick in ps.
- `rk4-fdm`, `rk4-fdm-fused` and `rk4-cuda` warn if `dt` may exceed a kinetic, reservoir diffusion or kinetic relaxation stability limit. The warning includes the estimated threshold in ps.
- A reservoir with two channels requires `spin: spinor`. With `reservoir_type: none`, a configured incoherent pump emits a warning because it has no reservoir to feed.

For feature compatibility and solver cost, see [Solver reference](solvers.md). For equations and the meaning of physical coefficients, see [Physics](../explanation/physics.md).
