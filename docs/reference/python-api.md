# Python API

Use `Config` and `SimulationController` for ordinary runs. The API entries below are generated from public objects and their docstrings; [Configuration](configuration.md) lists every field and [Plugins](../plugins/index.md) explains extension contracts.

`SimulationController` creates the grid, pump, potential, reservoir, boundary condition, state, model and solver, then records results while advancing to `solver.total_time` in ps. `run.set_initial_state(psi, reservoir_state=None, t=0.0)` accepts a field on the grid and optional reservoir arrays; `t` is the new origin in ps. `save_checkpoint(path)` and `load_checkpoint(path)` write and restore version 2 HDF5 state. `run.run_directory` identifies a created output directory. See [Results format](results-format.md) for saved shapes.

`polarism.compute_engine.compute_engine` exposes `xp`, the active NumPy or CuPy array namespace. `cfg.compute_engine.use_gpu` selects CuPy when available; plugin code should use backend arrays throughout each step. `to_cpu(value)` transfers a result to NumPy at an output boundary. See [Running on GPUs](../user-guide/running-on-gpus.md).

## Package exports

::: polarism
    options:
      members:
        - BoundaryCondition
        - Config
        - create_absorption_strategy
        - create_grid
        - create_potential
        - available_profiles
        - create_profile
        - register_profile
        - available_envelopes
        - create_envelope
        - register_envelope
        - create_reservoir
        - create_solver
        - AbstractSolver
        - Dispersion
        - Model
        - Term
        - StepReport
        - SolverCapabilities
        - LaserFactory
        - load_plugins
        - RealTimeVisualization
        - ResultNode
        - ResultProvider
        - SimulationController
        - Sweep
        - SweepResults
        - Results
        - load_results
        - SimulationGrid
        - SimulationState
        - Results2D
        - ResultScalar
        - ResultScalarGroup
        - ResultsManager
        - StorageVisitor
        - VisualizationVisitor
        - ResultVisitor

## Configuration dataclasses

::: polarism.config.simulation_parameters
    options:
      members:
        - GridParameters
        - PhysicsConstants
        - BoundaryConditionParameters
        - PotentialParameters
        - DriveParameters
        - NoiseParameters
        - InitialConditionParameters
        - LaserParameters
        - ReservoirParameters
        - SolverParameters
        - ResultParameters
        - ComputeEngineParameters
        - StopParameters
        - CheckpointParameters
        - SweepParameters
        - Config

## Parameter sweeps

::: polarism.sweep.Sweep

::: polarism.results.reader.SweepResults

::: polarism.results.accumulators.Accumulator

::: polarism.results.accumulators.AccumulatedResult

::: polarism.results.accumulators.register_accumulator

`polarism.results.available_accumulators` is the registry of available
accumulator classes.

::: polarism.analysis.g1

::: polarism.analysis.g2

## Controller and solver extension

::: polarism.simulation_controller.SimulationController

::: polarism.solver.AbstractSolver

::: polarism.model.Model

::: polarism.solver.StepReport

::: polarism.solver.SolverCapabilities

::: polarism.solver.register_solver

## Model component extension

The [laser](../plugins/lasers.md), [reservoir](../plugins/reservoirs.md),
[solver](../plugins/solvers.md), [term](../plugins/terms.md),
[dispersion](../plugins/dispersion.md), [grid](../plugins/grids.md), and
[absorption](../plugins/absorption.md) guides explain these bases and
registration decorators in context.

::: polarism.model_terms.Term

::: polarism.model_terms.register_term

::: polarism.dispersion.Dispersion

::: polarism.dispersion.register_dispersion

::: polarism.laser.abstract_laser.AbstractLaser

::: polarism.laser.register_laser

::: polarism.reservoir.abstract_reservoir.AbstractReservoir

::: polarism.reservoir.register_reservoir

::: polarism.potential.register_potential

::: polarism.grid.register_grid

::: polarism.boundary_conditions.absorption.register_absorption

## Batched state and reservoir API

`SimulationState(grid, eps=1e-3, precision="double", *, mode="complex_gaussian_zero_mean", k_cutoff_um=None, seed=None, batch_size=None, initial_condition=None, components=1)` creates seeded initial fields along the batch axis. `create_reservoir(reservoir_config, physics, grid, precision="double", *, batch_size=None, batch_physics=None)` creates matching zero density fields. Built-in reservoirs support `reset(batch_size=None)` to clear and resize their density fields. See [Batched simulations](../user-guide/solvers-and-time-stepping.md) for a working pattern.

## Compute engine

::: polarism.compute_engine.ComputeEngine
