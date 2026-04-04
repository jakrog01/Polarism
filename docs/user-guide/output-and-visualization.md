# Output and Visualization

Result handling is configured through `cfg.result`. The system supports live visualization plus batched storage in several formats.

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

## What gets recorded

The controller builds result nodes for quantities such as:

- condensate density `|psi|^2`
- total norm `N(t)`
- reservoir-derived outputs, when the reservoir exposes them
- pump field snapshots, when pump result exposure is enabled

## Pipeline outputs

The Slurm pipeline in `src/pump_multi_comparison/` writes a richer run directory that includes:

- configuration snapshots
- scenario metadata JSON files
- HDF5 simulation outputs
- per-scenario plots
- cross-scenario summary artifacts

Use the library storage path for single runs and the pipeline run directory for campaign-style sweeps.
