# API Overview

This section documents the public architecture of `polarism` at the level of modules, interfaces, factories, and configuration choices. It intentionally does not mirror the full source tree line by line.

The API reference is centered on the importable `polarism/` package. The `src/` tree contains example and HPC-pipeline code that uses `polarism`, but it is not part of the stable package surface.

## What is documented here

- top-level package entry points
- configuration blocks and supported options
- abstract interfaces such as solver, laser, reservoir, grid, and result provider
- factory-selection points where string options map to concrete implementations
- execution flow through controller, backend, and results management

## Architectural shape

At a high level, the package is organized like this:

1. `Config` collects all user-facing parameters.
2. Factories choose concrete grids, pumps, reservoirs, potentials, and solvers from those parameters.
3. `SimulationController` assembles the simulation objects and advances the run.
4. `ResultsManager` routes computed outputs to storage and visualization visitors.

## Documentation policy

This documentation prefers abstraction over implementation detail:

- describe responsibilities, not private helpers
- document supported options, not internal call graphs
- expose interfaces and selection points, not the entire codebase
