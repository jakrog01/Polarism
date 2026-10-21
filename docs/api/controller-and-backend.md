# Controller and Backend

## Simulation controller

`SimulationController` is the top-level runtime object. It assembles the configured components, advances time, and coordinates result production.

### Controller responsibilities

- create the simulation grid
- create the potential, pumps, reservoir, boundary condition, and solver
- initialize the condensate state
- run the timestep loop until `cfg.solver.total_time`
- dispatch result nodes through the results subsystem

### Controller contract

The controller is the right abstraction for running a simulation end-to-end. If you are not developing a new solver or component family, it should usually be your entry point.

### Checkpoint API

`set_initial_state(psi, reservoir_state=None, t=0.0)` starts from a supplied
real-space field at a solver step boundary. A leading batch axis is accepted;
the reservoir is reset to matching zero arrays if densities are omitted.

`save_checkpoint(path)` writes the current step, time, full-precision field,
authoritative spectral field when present, and reservoir arrays atomically to
HDF5. `load_checkpoint(path)` validates the grid shape, batch shape when an
initial state was explicitly set, reservoir type, and timestep before restoring
the fields. A precision difference casts the arrays with a warning. The loaded
step becomes the starting index for `run()` and `solver.total_time` remains the
absolute end time.

The `run_directory` property points to the unique directory for this run when
storage or checkpointing creates one. A resumed run gets its own directory.

## Compute engine

The compute engine abstracts the array backend used by the rest of the package.

### Role

- expose a NumPy-like namespace through `xp`
- switch between CPU and GPU execution
- keep solver, reservoir, and pump code backend-agnostic

### Main backend modes

| Mode | Trigger |
| --- | --- |
| CPU / NumPy | default path |
| GPU / CuPy | `cfg.compute_engine.use_gpu = True` and CuPy available |

### Why this matters

The compute-engine abstraction is what allows the same high-level simulation flow to run on laptop CPU paths and CUDA-oriented cluster runs without rewriting the model logic.
