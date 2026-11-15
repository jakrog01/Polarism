# Solver reference

Choose a solver from its supported physics and numerical cost. This matrix is read from the `SolverCapabilities` declarations of every registered built-in solver; plugin solvers add their own declarations.

`S` is `(nx,)` for 1D and `(ny,nx)` for 2D. Built-in solvers operate on canonical condensate fields `(B,C,*S)` and reservoir fields `(B,C_R,*S)`. `B` is the batch size. The `cuda` suffix identifies a GPU-optimized implementation; availability and CPU fallback vary by method. See [Running on GPUs](../user-guide/running-on-gpus.md).

## Declared capabilities

| Solver | Components | Dimensions | Grid boundaries | Reservoirs | Custom reservoirs | Adaptive |
| --- | --- | --- | --- | --- | --- | --- |
| `rk4-fdm` | 1, 2 | 1, 2 | closed-interval, periodic | single, double, quadratic-double, none | quantitative | no |
| `rk4-fdm-fused` | 1 | 1, 2 | closed-interval, periodic | single, double, quadratic-double | quantitative | no |
| `rk4-cuda` | 1, 2 | 1, 2 | closed-interval, periodic | single, double, quadratic-double, none | unsupported | no |
| `split-step-fft` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | no |
| `split-step-fft-cuda` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | no |
| `etd-rk2` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | no |
| `ip-rk4` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | no |
| `ifrk-fft-cuda` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | yes |
| `ifrk4-fft-cuda` | 1, 2 | 1, 2 | periodic | single, double, quadratic-double, none | quantitative | no |

All listed reservoir combinations are declared quantitative. `rk4-fdm-fused` excludes `none` and spinors. `rk4-cuda` rejects plugin reservoirs. Other methods declare quantitative support for them, although a plugin must implement the relevant reservoir interface correctly.

| Solver | Potential | Kinetic relaxation | Dispersion | Terms | Wigner noise | Coherent drive | Time-varying potential |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rk4-fdm` | yes | yes | parabolic | yes | yes | yes | yes |
| `rk4-fdm-fused` | yes | yes | parabolic | no | yes | no | no |
| `rk4-cuda` | yes | yes | parabolic | yes | yes | yes | yes |
| `split-step-fft` | yes | no | any registered | yes | yes | no | yes |
| `split-step-fft-cuda` | yes | no | parabolic | yes | yes | yes | yes |
| `etd-rk2` | yes | no | any registered | yes | yes | yes | yes |
| `ip-rk4` | yes | yes | any registered | yes | yes | yes | yes |
| `ifrk-fft-cuda` | yes | yes | parabolic | yes | yes | yes | yes |
| `ifrk4-fft-cuda` | yes | yes | parabolic | yes | yes | yes | yes |

All methods also accept `noise.stochastic: none`. `kinetic_relaxation_eta > 0` with a solver that does not implement it emits a warning: the term is ignored. On CUDA methods, a configured extra term needs a `CudaSnippet`, even when the capability table says terms are supported. Solvers using `{"*"}` for dispersion read any registered model; others accept only `parabolic`. Solver compatibility checks reject unsupported grid, component, reservoir, dispersion, term, coherent drive, potential envelope and noise combinations.

## Accuracy and cost

| Method | Time method and spatial operator | Cost and use |
| --- | --- | --- |
| `rk4-fdm` | Fourth-order RK4, finite-difference Laplacian | Reference CPU baseline; four stage evaluations per step and a restrictive explicit `dt` on fine grids. |
| `rk4-fdm-fused` | RK4, optimized finite differences | Lower overhead for scalar fields; fewer supported features. |
| `rk4-cuda` | RK4, tiled finite differences | GPU path for periodic or closed-interval grids; explicit stability limit remains. |
| `split-step-fft` | Second-order Strang splitting, FFT kinetic step | Periodic grid only; large steps can be efficient, but check splitting error. |
| `split-step-fft-cuda` | Second-order Strang splitting, cuFFT | Fast GPU path on periodic grids; scalar path keeps ψ in Fourier space between steps. |
| `etd-rk2` | Second-order exponential time differencing, FFT | Periodic grid; predictor-corrector reservoir coupling. |
| `ip-rk4` | Interaction-picture RK4, FFT | Periodic grid, broad physics support. |
| `ifrk-fft-cuda` | Lawson interaction-picture RK, FFT | Periodic grid; selectable order and adaptive substeps; more FFTs at higher stage counts. |
| `ifrk4-fft-cuda` | Fixed Lawson RK4 alias | Equivalent to `ifrk-fft-cuda` with `tableau: rk4`. |

Spatial discretization and time integration contribute separate errors. Check convergence in `grid.nx`, `grid.ny` and `solver.dt` in ps on representative physics. White-noise initial fields contain high wave numbers and can reduce the observed order of spectral Lawson methods. The explicit FDM methods warn when `dt` may exceed kinetic, diffusion or relaxation stability estimates. For a closed-interval grid, choose an FDM method; spectral methods require periodic boundaries. A periodic grid with a complex absorbing potential can model an open region.

## Lawson tableaus and adaptive stepping

| `solver.tableau` | Main order | Stages | Embedded error order | Adaptive allowed |
| --- | ---: | ---: | ---: | --- |
| `rk4` | 4 | 4 | none | no |
| `dp5` | 5 | 7 | 4 | yes |
| `dop853` | 8 | 12 | 5 and 3 | yes |
| `vern7` | 7 | 10 | 6 | yes |
| `vern9` | 9 | 16 | 8 | yes |

Only `ifrk-fft-cuda` declares adaptive stepping. Set `solver.adaptive = True` and choose an embedded tableau. `solver.dt` remains the sampling schedule tick in ps; the controller takes controlled substeps within it. Adaptive mode requires stochastic noise off, an absorption strategy with no after-step action, positive finite `rtol`, nonnegative finite `atol`, and a positive finite `dt_max` in ps when supplied. Single precision needs `rtol ≥ 1e-5`. Higher-order tableaus cost more stages and FFTs per step; they can save work for smooth fields and tight tolerances. Start with RK4 or the split-step method and measure convergence before choosing a larger tableau.

## RK4 CUDA laplacian

`rk4-cuda` reads `solver.laplacian`:

| Value | Stencil | Requirement |
| --- | --- | --- |
| `five-point` | Three-point in 1D, five-point in 2D | Default. |
| `isotropic-9pt` | Nine-point isotropic 2D Laplacian | Two dimensions and square cells, `dx = dy` in µm. |

The nine-point choice reduces direction-dependent spatial error. It represents the same kinetic operator. Compare grid convergence after changing the stencil. Other methods do not use this option.

## Batched runs

All built-in solvers accept a leading member axis. `rk4-fdm` and `rk4-fdm-fused` apply stencils on the last spatial axes. `split-step-fft`, `ip-rk4` and `etd-rk2` use batched FFT and array operations. `rk4-cuda` uses staged CUDA kernels across members; `split-step-fft-cuda` and the Lawson CUDA solvers use batched cuFFT and fused paths for supported built-in reservoirs. Plugin or diffusive reservoirs may take a generic path. Shared potential and pump arrays have shape `S`; member-varying inputs have shape `(B,1,*S)`. See [Parameter sweeps](../user-guide/parameter-sweeps.md) for batch setup.
