# Output and results

Select frame observables and a storage format before a run. Read saved arrays with `load_results` for plotting or further analysis.

## Save and read a run

```python
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = 64
cfg.grid.ny = 64
cfg.grid.lx = 60.0
cfg.grid.ly = 60.0
cfg.laser.laser_type = "continuous-gaussian"
cfg.laser.P0 = 0.3
cfg.laser.sigma_space = 6.0
cfg.solver.method = "ifrk-fft-cuda"
cfg.solver.tableau = "dop853"
cfg.solver.adaptive = True
cfg.solver.dt = 0.1
cfg.solver.total_time = 150.0
cfg.result.save_results = True
cfg.result.formats = ("hdf5",)
cfg.result.save_every = 1.0
cfg.result.observables = ("center-of-mass",)
controller = ps.SimulationController(cfg)
controller.run()
with ps.load_results(controller.run_directory) as result:
    density = result.fields["density"][:]
    norm = result.scalars["norm"][:]
    x, y, t = result.x, result.y, result.time
```

Here `x` and `y` are in µm, `t` is in ps, density is in µm⁻², and norm is dimensionless particle number. The initial state is recorded at its starting time. `save_every` is in ps and must be a positive multiple of `solver.dt`. If unset, frames are recorded every step. Each run gets a timestamped directory under `result.output_directory`.

## Frame quantities

Every recorded frame includes `density` and `norm`. Set `result.save_psi` to retain the complex field for later phase analysis. Pump and reservoir fields are exposed when their component's `expose_results` setting permits it. The coherent drive adds a static `drive` field when enabled. Spinor runs also expose component densities, Stokes fields, and degree of polarization. Laser peak powers appear in the `P_lasers` scalar group. For exact dataset names, axes, units, and on-disk layout, see [Results format](../reference/results-format.md).

Choose extra frame observables with `result.observables`:

| Name | Output | Meaning |
| --- | --- | --- |
| `center-of-mass` | `center_x`, `center_y` in µm | Density-weighted position; `center_y` exists only in 2D |
| `momentum-density` | `momentum_density` | FFT-shifted instantaneous momentum density |
| `current` | `current_x`, `current_y` | Probability current; `current_y` exists only in 2D |
| `angular-momentum` | `angular_momentum` in units of ħ per particle | Orbital angular momentum about the grid origin; 2D only |
| `vortices` | `vortex_charge`, `vortex_count` | Winding per plaquette and number of resolved vortices; 2D only |

Spinor center-of-mass and angular-momentum outputs have `_plus` and `_minus` suffixes. The current uses `(ħ/m_eff) Im(ψ*∇ψ)`. `vortex_charge` uses phase winding around each plaquette; `vortex_count` ignores plaquettes whose corner densities are all below `10⁻³` of the peak density. A single plaquette may not resolve a charge-two vortex. For time averages, correlations, and spectra, use [Observables and spectra](observables-and-spectra.md). Custom observables and visitors are described in [Observable plugins](../plugins/observables.md).

## Storage choices

Use `result.formats` to select `hdf5` for long runs and lazy frame reads, `npy` for NumPy-based processing, or `json` for small debugging runs. HDF5 is uncompressed by default. `result.compression` accepts `none`, `lzf`, or `gzip`; compression saves disk space at the cost of write time. `result.save_psi` can greatly increase output size.

Storage queues device frames and writes them on a background thread. When the bounded queue fills, integration waits for storage. A storage error stops the run. `load_results` supports saved HDF5, NPY, and JSON results; HDF5 slices load only requested frames. Close a result or use a context manager to release file handles. A saved sweep returns a `SweepResults` view; see [Parameter sweeps](parameter-sweeps.md).

## Live plots and movies

Set `result.real_time_view: true` for an interactive 2D plot. `result.real_time_refresh_interval` is in simulation ps between display updates. For a 2D movie, set `result.animate: true`, choose `result.animation_fields`, and set `result.animation_output` to an MP4 path if you want a specific location. `animation_fps` is playback frames per second; `animation_target_seconds` controls target movie length. The encoder is an external ffmpeg process. See [Configuration reference](../reference/configuration.md) for output options.

A custom `ResultVisitor` can be added with `controller.add_visitor(visitor)` to process each recorded frame. See [Observable plugins](../plugins/observables.md) for the visitor interface.
