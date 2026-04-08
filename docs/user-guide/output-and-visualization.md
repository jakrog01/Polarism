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

## Pipeline outputs

The Slurm pipeline in `src/pump_multi_comparison/` writes a richer run directory that includes:

- configuration snapshots
- scenario metadata JSON files
- HDF5 simulation outputs
- per-scenario plots
- cross-scenario summary artifacts

Use the library storage path for single runs and the pipeline run directory for campaign-style sweeps.
