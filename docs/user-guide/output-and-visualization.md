# Output and Visualization

Result handling is configured through `cfg.result`. A run can write result
files, stream a movie, show a live plot, or feed custom visitors; all of them
receive the same result frames.

## What gets recorded

The controller builds one result node per quantity. Saved names are ASCII;
plots and movie titles use the display label.

| Name | Label | Kind | Present when |
| --- | --- | --- | --- |
| `density` | `|ψ|²` | field | always |
| `norm` | `N(t)` | scalar | always |
| `psi` | `psi` | complex field | `cfg.result.save_psi = True` |
| `pump` | `Pump` | field | `cfg.laser.expose_results = True` (default) |
| `N_R`, `N_R_max` | | field, scalar | `single` reservoir |
| `nA`, `nI`, `nA_max`, `nI_max` | | fields, scalars | `double` reservoir |
| `nR`, `nI`, `nR_max`, `nI_max` | | fields, scalars | `quadratic-double` reservoir |

Per-laser peak powers are stored as the scalar group `P_lasers` (`L0`, `L1`, …).
Additional quantities come from [observable plugins](../plugins/observables.md)
selected with `cfg.result.observables`.

The initial state is recorded at its starting time `t0`. After that, every
output receives a frame every `save_every` ps. The interval must be a positive
multiple of `solver.dt`; by default every step is saved.

## Saving results

```python
cfg.result.save_results = True
cfg.result.formats = ("hdf5",)
cfg.result.save_every = 0.01
cfg.result.save_psi = False
cfg.result.compression = "none"
cfg.result.output_directory = "simulation_results"
```

Each run writes into its own directory `<output_directory>/<YYYYmmdd-HHMMSS>`;
a numeric suffix (`-1`, `-2`, …) is added if that name already exists.

Built-in formats:

| Format | Files | Use |
| --- | --- | --- |
| `hdf5` | `results.h5` | default choice for long runs and large fields |
| `npy` | `batch_*.npz`, `static_*.npy`, `grid_*.npy`, `metadata.json` | NumPy-only downstream tools |
| `json` | `batch_*.json`, `metadata.json` | debugging small runs |

The older switches `save_hdf5`, `save_npy` and `save_json` still work and are
merged with `formats`. Further formats can be added as
[storage plugins](../plugins/storage.md).

`save_psi = True` stores the complex condensate field, which keeps the phase
needed to locate vortices or to compute momentum-space quantities later. JSON
stores complex arrays as `{"real": ..., "imag": ...}`.

### Performance

Storage copies each frame on the device and returns immediately; a background
thread moves the data to host memory and writes it, so the solver keeps
running while files are written. The queue of pending frames is bounded by
frame count and by device memory, and the simulation waits if the disk cannot
keep up. Pending frames are written when the run ends or is interrupted, and a
storage error stops the run.

HDF5 datasets are uncompressed by default. Simulation fields compress poorly
(typically to 60–90 % of their size) while compression multiplies the write
time roughly tenfold, so set `compression = "lzf"` or `"gzip"` (with
`compression_level` from 0 to 9) only when disk space matters more than run
time. Compressed HDF5 datasets use the shuffle filter; NPY batches use
`numpy.savez_compressed` when any compression is selected.

A pump that does not change in time (continuous lasers without a positive
`delay`) is written once as a static field: `static/pump` in HDF5,
`static_pump.npy` in NPY, or under `"static"` in the first JSON batch.

## Reading results

`polarism.load_results` reads HDF5 and NPY runs with the same API:

```python
import polarism as ps

with ps.load_results("simulation_results/20260101-120000") as result:
    density = result.fields["density"][::10]
    norm = result.scalars["norm"][:]
    pump = result.static["pump"][:]
    x, y, t = result.x, result.y, result.time
    dt = result.config["solver"]["dt"]
```

For HDF5 runs, `fields` returns lazy scalar-field views and `scalars` and
`static` return `h5py` datasets. Slicing reads only the selected frames from
disk; use `[:]` for the whole array. NPY and JSON runs return NumPy arrays.
`attrs` holds file metadata including `format_version` and `polarism_version`.
Only format v3 files load.

For saved parameter sweeps, `polarism.load_results` accepts the sweep directory
and returns a `SweepResults` view with lazy frame and member slicing. See
[Parameter Sweeps](parameter-sweeps.md#output-and-restart) for its layout and
reader API.

## File formats

The HDF5 results layout (`format_version = 3`) is:

| Path | Content |
| --- | --- |
| `time` | frame times in ps |
| `fields/<name>` | field frames, shape `(time, component, *S)` for one run or `(time, member, component, *S)` for a batch |
| `scalars/<name>` | scalar series |
| `scalar_groups/<group>/<label>` | grouped scalar series |
| `static/<name>` | time-independent fields |
| `grid/x` and, in 2D, `grid/y` | spatial grid coordinates in µm |
| `config` | attribute `json` with the full configuration |

The root stores `ndim`, `grid_type`, `components = ["scalar"]`,
`polarism_version`, and coordinate and time units. Every dataset has `dims`
and `units` attributes. Coordinates use µm and time uses ps. Density has units
µm⁻ⁿᵈⁱᵐ, wavefunction has µm⁻ⁿᵈⁱᵐᐟ², and potential has meV; unknown units are
an empty string. Static fields omit the time dimension. In 1D, HDF5 stores
`grid/x` without `grid/y`. `load_results` hides the scalar component axis;
the public field shape remains `(time, *S)` or `(time, member, *S)`.

NPY and JSON fields keep those public shapes. Their `metadata.json` records
`format_version = 3`, `ndim`, and each array's `dims` and `units`. Older result
versions are rejected with the found and supported versions. Live visualization
and animation are currently available only in 2D.

## Online animation

The core library can stream a movie while the simulation is running:

```python
cfg.result.animate = True
cfg.result.animation_fps = 8
cfg.result.animation_target_seconds = 60
cfg.result.animation_fields = ("density", "nA", "nI", "pump")
cfg.result.animation_backend = "auto"
cfg.result.animation_encoder = "h264_nvenc"
cfg.result.animation_output = "simulation_results/animation.mp4"
```

Panels are titled with the display labels. The 0.3 names (`|ψ|²`, `N(t)`,
`Pump`) are still accepted in `animation_fields` with a deprecation warning.

The animation visitor receives backend-native arrays, so a CuPy run renders
false-colour panels on the GPU; only the final RGB frame goes to ffmpeg.
A broken encoder is aborted and the animation is disabled, while storage
failures stay fatal, so a movie problem never costs the simulation data.

Colour limits are inferred from the first sample frames. For pulsed or
delayed-growth runs, fixed limits avoid early low-density frames making later
frames saturate.

## Live visualization

```python
cfg.result.real_time_view = True
cfg.result.real_time_refresh_interval = 0.1
```

This is useful for interactive debugging, not for unattended cluster runs.

## Custom visitors

`SimulationController.add_visitor(visitor)` adds any
[`ResultVisitor`](../plugins/observables.md); it receives every
recorded frame, and its `finalize()` is called when the run ends.
