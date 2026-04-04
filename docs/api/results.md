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

## Why this abstraction matters

The physics and solver layers do not need to know whether a quantity will be plotted, written to HDF5, or ignored. They only expose result nodes. That keeps numerical code separate from output boilerplate.
