# Top-Level Package

The top-level `polarism` package is designed as a compact orchestration surface over a set of interchangeable simulation components.

## Main public entry points

| Symbol | Role |
| --- | --- |
| `Config` | Root dataclass holding all simulation parameters |
| `SimulationController` | High-level execution object that assembles and runs a simulation |
| `create_solver` | Selects a solver backend from `cfg.solver.method` |
| `create_potential` | Builds the external potential from `cfg.potential` |
| `create_reservoir` | Builds the reservoir model from `cfg.reservoir` |
| `LaserFactory` | Builds one or more pump profiles from `cfg.laser` |
| `BoundaryCondition` | Wraps boundary absorption logic around each solver step |
| `SimulationState` | Holds evolving field state such as `psi` |
| `ResultsManager` | Sends registered result nodes to active visitors |

## Design intent

The package-level API is intentionally small:

- physics and numerics live in dedicated modules
- object selection is driven by config and factory methods
- higher-level workflows should consume `polarism` as a library rather than patching solver internals directly

## What is not part of this package surface

The following belong outside the core package API:

- Slurm submission logic
- campaign orchestration
- study-specific threshold search
- cluster-specific environment wrappers

Those are documented in the `Examples` section instead.
