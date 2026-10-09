# Output and Visualization

Result handling is configured through `cfg.result`. The core library supports live visualization plus batched storage in several formats, plus append-only HDF5 helpers for large batch runs.

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

## Appendable HDF5 output

Batch pipelines that stream explicit field snapshots do not use `cfg.result.save_hdf5`. Instead, they write scenario outputs through the appendable HDF5 utilities exported from `polarism.results.storage`.

That path now behaves as follows:

- `create_hdf5_writer(...)` selects a CPU-buffered writer on CPU runs and a GPU-async writer on CuPy runs.
- the GPU writer uses double buffering and attempts pinned host-memory registration so device-to-host copies can overlap with simulation work
- `compute_batch_size(...)` derives the batch depth from the active field set, spatial shape, and currently free GPU memory, while reserving headroom for the solver
- on CPU runs the heuristic falls back to a fixed batch depth of `500`

This separation keeps the reusable storage implementation in `polarism.results.storage` while the pipeline remains responsible only for scenario orchestration and field/scalar selection.
