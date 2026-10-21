# Results

The results subsystem is organized around two abstractions: providers expose quantities, and visitors decide what to do with them.

## Result provider interface

Objects that want to expose outputs implement the `ResultProvider` contract by returning a list of result nodes.

Typical providers include:

- reservoir models
- potential components
- simulation state related outputs

## Results manager

`ResultsManager` collects nodes and forwards them to registered visitors at each output step.

### Responsibilities

- maintain the active list of result nodes
- cache computed values for one timestep
- send those values to every visitor

The manager is backend-aware.  CPU visitors receive NumPy arrays, while visitors
that declare `data_location = "device"` receive backend-native arrays, such as
CuPy arrays in GPU runs.  Nodes requested only by device visitors are evaluated
without scalar reductions, which avoids unnecessary GPU synchronisation.

## Visitor model

Visitors interpret result nodes in different ways:

- storage visitors write data to disk
- visualization visitors render plots or live views
- additional visitors can post-process results without changing solver code

Each visitor can also control how failures are handled:

| Attribute | Default | Meaning |
| --- | --- | --- |
| `data_location` | `"cpu"` | receive CPU arrays unless explicitly set to `"device"` |
| `fatal_on_error` | `True` | propagate visitor exceptions and stop the run |

`AnimationVisitor` is the main non-fatal visitor.  It sets
`fatal_on_error = False`, so an encoder failure aborts and disables only the
animation stream.  Storage visitors keep the default fatal behaviour; failed
HDF5, JSON, or NPY output still stops the simulation instead of silently losing
data.

## Online animation

`polarism.results.visitors.AnimationVisitor` streams field frames during the
simulation instead of reading HDF5 after the run.  It is designed for large GPU
runs where post-hoc frame rendering can dominate wall time.

The animation path has three layers:

| Layer | Module | Role |
| --- | --- | --- |
| panel rendering | `polarism.results.rendering.gpu_movie` | apply transforms, normalize, map to LUT colours, tile panels |
| encoding | `polarism.results.rendering.video_encoder` | stream raw RGB frames to ffmpeg with encoder fallback |
| visitor | `polarism.results.visitors.animation_visitor` | collect frames from `ResultsManager` or direct pipeline calls |

Supported field transforms include:

- `abs2` for condensate density from complex `psi`
- `kspace_log` for normalized log k-space power
- `None` for already-real fields

The encoder path validates explicit PyNvVideoCodec requests as unsupported for
now.  `backend="auto"` and `backend="ffmpeg"` use ffmpeg encoders, preferring
GPU-capable NVENC when available and falling back to CPU encoders when needed.

## Output backends

The user-facing result configuration exposes these main storage choices:

| Flag | Meaning |
| --- | --- |
| `save_hdf5` | structured binary storage for large runs |
| `save_json` | lightweight metadata-oriented serialization |
| `save_npy` | NumPy array dumps |
| `real_time_view` | interactive visualization during a run |
| `animate` | stream an online animation during the run |

The reusable storage module also exports append-only HDF5 helpers used by batch pipelines:

| Symbol | Role |
| --- | --- |
| `AppendableHDF5Writer` | abstract append-only writer contract |
| `CpuBufferedHDF5Writer` | CPU-oriented batched HDF5 writer |
| `GpuAsyncHDF5Writer` | GPU-oriented double-buffered HDF5 writer |
| `compute_batch_size` | GPU-memory-aware batch-depth heuristic |
| `create_hdf5_writer` | backend-selection factory for CPU vs GPU runs |

These helpers are separate from the visitor-based `cfg.result` path.

The storage visitor snapshots backend arrays when a frame is visited and sends
them to one background writer. A bounded queue limits pending device snapshots
to at most about 1 GiB or one quarter of free device memory. The writer waits
for the CUDA event before transfer, preserves frame order, and reports write
errors to the simulation thread. Finalization drains pending frames.

HDF5 datasets are uncompressed by default. `cfg.result.compression` also accepts
`"lzf"` and `"gzip"` (with the shuffle filter); `compression_level` controls gzip only. The pump is
stored once under `static/Pump` when every laser is time independent and has
no positive delay. NPY writes `static_Pump.npy`, and JSON puts the field under
`"static"` in the first batch. `save_psi=True` adds a complex `psi` field;
JSON represents each complex array with `"real"` and `"imag"` arrays.

## Why this abstraction matters

The physics and solver layers do not need to know whether a quantity will be plotted, written to HDF5, or ignored. They only expose result nodes. That keeps numerical code separate from output boilerplate.
