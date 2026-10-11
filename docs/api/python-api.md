# Python API

These entries are generated from public Python objects and their docstrings. The [hand-written reference](overview.md) explains configuration choices and runtime behavior.

For constructor calls and working extension examples, see the [plugin guide](../plugins/index.md).

## Package exports

::: polarism
    options:
      members:
        - BoundaryCondition
        - Config
        - create_potential
        - create_reservoir
        - create_solver
        - AbstractSolver
        - RK4CudaSolver
        - LaserFactory
        - load_plugins
        - RealTimeVisualization
        - ResultNode
        - ResultProvider
        - SimulationController
        - SimulationGrid2D
        - SimulationState
        - Results2D
        - ResultScalar
        - ResultScalarGroup
        - ResultsManager
        - StorageVisitor
        - VisualizationVisitor
        - ResultVisitor
        - hdf5_storage
        - json_storage
        - npy_storage
        - base_storage

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
        - Config

## Controller and solver extension

::: polarism.simulation_controller.SimulationController

::: polarism.solver.AbstractSolver

::: polarism.solver.register_solver

## Model component extension

The [laser](../plugins/lasers.md), [reservoir](../plugins/reservoirs.md),
[solver](../plugins/solvers.md), [grid](../plugins/grids.md), and
[absorption](../plugins/absorption.md) guides explain these bases and
registration decorators in context.

::: polarism.laser.abstract_laser.AbstractLaser

::: polarism.laser.register_laser

::: polarism.reservoir.abstract_reservoir.AbstractReservoir

::: polarism.reservoir.register_reservoir

::: polarism.potential.register_potential

::: polarism.grid.register_grid

::: polarism.boundary_conditions.absorption.register_absorption
