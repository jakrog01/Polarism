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

## Boundary-condition options

| Field | Meaning | Main options |
| --- | --- | --- |
| `absorption` | absorbing-boundary strategy | `no-absorption`, `mask`, `cap` |
| `profile_type` | absorption profile shape | profile-dependent string |
| `strength` | boundary damping strength | positive float |
| `mask_width_percent` | boundary-layer width as fraction of domain | float in `(0, 1)` |

## Potential options

| Field | Meaning | Main options |
| --- | --- | --- |
| `potential_type` | external potential model | `zero`, `double-well-supergaussian` |
| `x1`, `y1`, `x2`, `y2` | feature positions for structured potentials | floats |
| `V1`, `V2` | potential amplitudes | floats |
| `w1`, `w2` | characteristic widths | positive floats |
| `order` | super-Gaussian order | positive float |

## Laser options

| Field | Meaning | Main options |
| --- | --- | --- |
| `mode` | single or multi-pump setup | `single`, `multiple` |
| `config_file` | YAML file for multi-pump setups | path string |
| `laser_type` | pump-profile selection | `uniform`, `continuous-gaussian`, `continuous-exp`, `pulse-gaussian` |
| `P0`, `Pmax` | pump amplitudes | floats |
| `x0`, `y0` | pump center | floats |
| `sigma_space` | spatial width scale | positive float |
| `sigma_time` | pulse width in time | positive float |
| `pulse_separation` | distance between pulses | positive float |
| `cutoff_sigma` | pulse truncation radius in sigma units | positive float |
| `delay` | start delay before pump turns on | nonnegative float |

## Reservoir options

| Field | Meaning | Main options |
| --- | --- | --- |
| `reservoir_type` | reservoir-model selection | `single`, `double` |

## Solver options

| Field | Meaning | Main options |
| --- | --- | --- |
| `method` | time integrator | `rk4-fdm`, `rk4-fdm-fused`, `rk4-cuda`, `rk4-cuda-v100`, `split-step-fft`, `etd-rk2`, `ip-rk4` |
| `dt` | timestep | positive float |
| `total_time` | total simulation time | positive float |
| `precision` | arithmetic mode where supported | typically `single`, `double` |

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

## Compute-engine options

| Field | Meaning | Main options |
| --- | --- | --- |
| `use_gpu` | request CuPy/CUDA backend | boolean |
| `gpu_device` | CUDA device index | nonnegative integer |

## Physics block

The `physics` block stores the scalar coefficients entering the condensate and reservoir equations, including:

- `hbar`, `m_eff`
- `gamma_C`, `gamma_R`, `gamma_I`, `gamma_A`
- `g_C`, `g_R`
- `R`, `R_IA`, `R_AI`
- `init_eps`

These values are model parameters, not high-level feature flags, so they should be changed with physical units and stability constraints in mind.
