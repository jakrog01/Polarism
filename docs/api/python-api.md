# Python API

These entries are generated from public Python objects and their docstrings. The [hand-written reference](overview.md) explains configuration choices and runtime behavior.

For constructor calls and working extension examples, see the [plugin guide](../plugins/index.md).

## Package exports

::: polarism
    options:
      members:
        - BoundaryCondition
        - Config
        - create_absorption_strategy
        - create_grid
        - create_potential
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
        - LaserParameters
        - ReservoirParameters
        - SolverParameters
        - ResultParameters
        - ComputeEngineParameters
        - SweepParameters
        - Config

## Parameter sweeps

::: polarism.sweep.Sweep

::: polarism.results.reader.SweepResults

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

`SimulationState(grid, eps=1e-3, precision="double", *, mode="complex_gaussian_zero_mean", k_cutoff_um=None, seed=None, batch_size=None)` creates identical seeded initial fields along the batch axis. `create_reservoir(reservoir_config, physics, grid, precision="double", *, batch_size=None)` creates matching zero density fields. Built-in reservoirs support `reset(batch_size=None)` to clear and resize their density fields. See [Batched simulations](../user-guide/time-evolution.md#batched-simulations) for a working pattern.
