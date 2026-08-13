# Option Matrix

This page collects the main configuration choices in one place. It is meant as a fast selection guide, not a replacement for the detailed user guide.

## Grid topology

| Option | Use when | Notes |
| --- | --- | --- |
| `periodic` | the simulated domain should wrap around naturally, or you want the cleanest match to FFT-based solvers | best fit for spectral methods such as `split-step-fft` and `etd-rk2` |
| `closed-interval` | you want a bounded box with explicit edges | usually pairs most naturally with FDM-based solvers |

## Boundary absorption

| Option | Use when | Notes |
| --- | --- | --- |
| `no-absorption` | you want no artificial damping at the domain edge | best only when edge reflections are physically acceptable |
| `mask` | you want a simple absorbing rim near the boundary | practical default for many finite-domain runs |
| `cap` | you want absorption represented as a complex absorbing potential | useful when you want boundary damping folded into the effective potential picture |

## Potential type

| Option | Use when | Notes |
| --- | --- | --- |
| `zero` | you want a homogeneous system without external trapping | simplest baseline for convergence and solver comparison |
| `double-well-supergaussian` | you want two localized wells with smooth but tunable walls | exposes positions, amplitudes, widths, and super-Gaussian order |

## Laser type

| Option | Use when | Notes |
| --- | --- | --- |
| `uniform` | the pump should be spatially homogeneous | simplest driven case |
| `continuous-gaussian` | you want a localized continuous spot | common default for a single focused pump |
| `continuous-exp` | you want a localized pump with longer radial tails than a Gaussian | matches the current implementation exactly |
| `continuous-exp-length` | you want an exponential pump whose `sigma_space` is the 1/e radial decay length | preferred for new length-consistent configurations |
| `pulse-gaussian` | you want repeated pulsed excitation with Gaussian spot shape | supports per-pulse strength ramping from `P0` up to `Pmax`; choose `power_definition` carefully |

For pulsed Gaussian studies:

| `power_definition` | Use when | Notes |
| --- | --- | --- |
| `peak_amplitude` | reproducing legacy fixed-spot runs | fixed `P0` means fixed centre density, not fixed total dose |
| `pulse_energy` | comparing spot sizes or geometries | fixed `P0` means fixed integrated pulse dose over space and time |

## Laser mode

| Option | Use when | Notes |
| --- | --- | --- |
| `single` | one pump definition is enough | configure directly on `cfg.laser` |
| `multiple` | you want several pumps in one run | define them in an external YAML file |

## Reservoir model

| Option | Use when | Notes |
| --- | --- | --- |
| `single` | one effective active reservoir is enough for the physics you want to resolve | cleanest baseline model |
| `double` | you need to separate inactive and active reservoir populations | adds inter-reservoir transfer timescales and more stiffness risk |
| `quadratic-double` | you model pulsed nonresonant excitation with delayed inactive-to-active feeding | uses `kappa*nI^2` transfer and can retain pulse-train memory |

## Solver method

| Option | Use when | Notes |
| --- | --- | --- |
| `rk4-fdm` | you want the safest reference solver | best first choice for validation and comparisons |
| `rk4-fdm-fused` | you want a faster FDM-style path without changing the basic method family | good follow-up once `rk4-fdm` is validated |
| `rk4-cuda` | you want FDM GPU production runs on periodic or closed-interval grids | FDM production/reference; stage-coupled quadratic-double |
| `rk4-cuda-v100` | you specifically target V100 hardware | hardware-tuned CUDA variant; numerically identical to `rk4-cuda` |
| `ifrk4-fft-cuda` | you want GPU-native spectral production on a periodic grid | requires `periodic`; use CAP absorber for open boundaries; stage-coupled quadratic-double |
| `split-step-fft` | diagnostic spectral reference on simple periodic setups | `closed-interval` path uses CPU DCT with host-device copies — not GPU production |
| `etd-rk2` | low-order exponential time-differencing on a periodic grid | periodic only; reservoir not stage-coupled; diagnostic use |
| `ip-rk4` | spectral-style RK4 reference | `closed-interval` path uses CPU DCT with host-device copies — not GPU production |

## RK4 CUDA Laplacian

| Option | Use when | Notes |
| --- | --- | --- |
| `five-point` | you want historical `rk4-cuda` behavior | fastest and backward-compatible |
| `isotropic-9pt` | diagonal/axis artifacts suggest stencil anisotropy | requires square cells and should still be grid-checked |

## Initial seed mode

| Option | Use when | Notes |
| --- | --- | --- |
| `legacy_positive_uniform` | you need to reproduce old runs | biased positive-uniform complex seed |
| `complex_gaussian_zero_mean` | you want an unbiased stochastic seed | no spectral cutoff |
| `filtered_complex_gaussian` | you want an explicit condensate-band seed cutoff | requires `init_k_cutoff_um`; report/sweep the cutoff |

## Backend choice

| Option | Use when | Notes |
| --- | --- | --- |
| `use_gpu = False` | you want the most portable CPU baseline | best default for correctness checks |
| `use_gpu = True` | CuPy is available and you want GPU execution | validate against a CPU or FDM baseline first |

## Practical default path

If you want a stable starting point, the shortest conservative path is:

1. `grid_type = "periodic"` or `closed-interval`, depending on the physical boundary you want
2. `potential_type = "zero"` unless you explicitly need trapping
3. `laser_type = "continuous-gaussian"`
4. `reservoir_type = "single"`
5. `solver.method = "rk4-fdm"`
6. `use_gpu = False`

After that baseline is validated, move to more specialized solvers, richer reservoir models, or cluster-oriented example workflows.
