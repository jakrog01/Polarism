# Output and Visualization

Result handling is configured through `cfg.result`. The core library supports live visualization plus batched storage in several formats, while the Slurm pipeline adds its own append-only HDF5 path for large scenario runs.

## Live visualization

Enable local plotting with:

```python
cfg.result.real_time_view = True
cfg.result.real_time_refresh_interval = 0.1
```

This is useful for interactive debugging, not for large unattended cluster runs.

## Persistent storage

The main storage switches are:

```python
cfg.result.save_results = True
cfg.result.save_hdf5 = True
cfg.result.save_json = False
cfg.result.save_npy = False
cfg.result.save_interval = 10
cfg.result.batch_size = 1000
cfg.result.output_directory = "simulation_results"
```

Format guidance:

- `save_hdf5`: best default for long runs and large field data.
- `save_npy`: useful when you want lightweight NumPy reloads for downstream analysis.
- `save_json`: useful for debugging small outputs, but not efficient for large fields.
- `batch_size`: controls how many result frames the library visitors buffer before flushing to disk.

## Online animation

The core library can stream a movie while the simulation is running:

```python
cfg.result.animate = True
cfg.result.animation_fps = 8
cfg.result.animation_target_seconds = 60
cfg.result.animation_fields = ("|ψ|²", "nA", "nI", "Pump")
cfg.result.animation_backend = "auto"
cfg.result.animation_encoder = "h264_nvenc"
cfg.result.animation_output = "simulation_results/animation.mp4"
```

The animation visitor receives backend-native arrays from `ResultsManager`, so a
CuPy run can render false-colour panels without first converting every field to
CPU arrays for the visitor.  The final encoded frame still passes through
ffmpeg as raw RGB video.

Animation failures are treated differently from storage failures.  A broken
encoder is aborted and the animation visitor is disabled, while HDF5/JSON/NPY
storage failures remain fatal.  This keeps simulation data trustworthy while
preventing a movie encoder crash from destroying a long run.

Colour limits are inferred from the first sample frames unless fixed limits are
provided by the caller.  For pulsed or delayed-growth runs, fixed limits are
recommended so that early low-density frames do not make later frames saturate.

## What gets recorded

The controller builds result nodes for quantities such as:

- condensate density `|psi|^2`
- total norm `N(t)`
- reservoir-derived outputs, when the reservoir exposes them
- pump field snapshots, when pump result exposure is enabled

## Appendable HDF5 pipeline output

The example pipeline in `src/pump_multi_comparison/` does not use `cfg.result.save_hdf5`. Instead, it writes scenario outputs through the appendable HDF5 utilities exported from `polarism.results.storage`.

That path now behaves as follows:

- `create_hdf5_writer(...)` selects a CPU-buffered writer on CPU runs and a GPU-async writer on CuPy runs.
- the GPU writer uses double buffering and attempts pinned host-memory registration so device-to-host copies can overlap with simulation work
- `compute_batch_size(...)` derives the batch depth from the active field set, spatial shape, and currently free GPU memory, while reserving headroom for the solver
- on CPU runs the heuristic falls back to a fixed batch depth of `500`

This separation keeps the reusable storage implementation in `polarism.results.storage` while the pipeline remains responsible only for scenario orchestration and field/scalar selection.

## Pipeline animations

`src/pump_multi_comparison/` renders scenario movies from the scratch-local HDF5
file after the numerical simulation finishes.  When `output.render_animation:
true`, each scenario job:

- resolves the ffmpeg binary from `FFMPEG_BIN` or `PATH`
- respects `RENDER_ENCODER` when it is set
- scans all recorded HDF5 frames to compute global colour limits
- uses physical zero as the black point for non-negative fields
- streams rendered RGB frames to ffmpeg without loading the whole movie into RAM
- writes `animation_status` and `animation_error` into scenario metadata

If animation is required and fails after simulation data has been written, the
scenario copies back HDF5, sidecar, PNG, and metadata artifacts first, then exits
nonzero.  This makes a failed movie visible to Slurm without discarding the
scientific outputs.

The default movie panels are `psi` as `|ψ|²`, `nA`, and `nI`.  `Pump` is kept in
static PNGs and scalar traces rather than the default movie because narrow pulse
trains can be under-sampled by the field-record cadence.

Automatic global colour limits are the default.  Set per-field limits only when
multiple movies must share an identical visual scale:

```yaml
output:
  render_animation: true
  animation_clim:
    psi: [0.0, 1.0]
    nA: [0.0, 10.0]
    nI: [0.0, 100.0]
```

Malformed `animation_clim` entries are ignored with a warning.

## Pipeline outputs

The Slurm pipeline in `src/pump_multi_comparison/` writes a richer run directory that includes:

- configuration snapshots
- scenario metadata JSON files
- HDF5 simulation outputs
- per-scenario plots
- optional `dynamics.mp4` or `dynamics.mkv` movies
- cross-scenario summary artifacts

For pulsed Gaussian runs, scalar sidecars include both local pump strength and
integrated pump-dose diagnostics:

| Scalar | Meaning |
| --- | --- |
| `P_max` | maximum local pump value on the grid at the recorded time |
| `P_area_integral` | instantaneous spatial integral of the pump field, \(\sum P\,dx\,dy\) |
| `P_cumulative_area_time_integral` | cumulative delivered dose, accumulated every time step as \(P_{\mathrm{area}}\,dt\) |

When `power_definition: pulse_energy` is used, `P_max` changes with
`sigma_space`, while `P_area_integral` and its cumulative integral are the
quantities that verify fixed total pulse dose.

Use the library storage path for single runs and the pipeline run directory for campaign-style sweeps.
