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

## Visitor model

Visitors interpret result nodes in different ways:

- storage visitors write data to disk
- visualization visitors render plots or live views
- additional visitors can post-process results without changing solver code

## Output backends

The user-facing result configuration exposes these main storage choices:

| Flag | Meaning |
| --- | --- |
| `save_hdf5` | structured binary storage for large runs |
| `save_json` | lightweight metadata-oriented serialization |
| `save_npy` | NumPy array dumps |
| `real_time_view` | interactive visualization during a run |

The reusable storage module also exports append-only HDF5 helpers used by the example pipeline:

| Symbol | Role |
| --- | --- |
| `AppendableHDF5Writer` | abstract append-only writer contract |
| `CpuBufferedHDF5Writer` | CPU-oriented batched HDF5 writer |
| `GpuAsyncHDF5Writer` | GPU-oriented double-buffered HDF5 writer |
| `compute_batch_size` | GPU-memory-aware batch-depth heuristic |
| `create_hdf5_writer` | backend-selection factory for CPU vs GPU runs |

These helpers are separate from the visitor-based `cfg.result` path. They are intended for workflows that stream explicit field snapshots, such as `src/pump_multi_comparison/`.

## Why this abstraction matters

The physics and solver layers do not need to know whether a quantity will be plotted, written to HDF5, or ignored. They only expose result nodes. That keeps numerical code separate from output boilerplate.
