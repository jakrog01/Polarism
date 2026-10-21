# Plugins

A plugin is a Python module that registers a laser, potential, reservoir,
solver, grid, absorption strategy, storage format, or observable under a configuration name. Start with
[Your first plugin](first-plugin.md) for a complete run before using the
component-specific contracts below.

## How discovery works

- Place `.py` files under `<current working directory>/polarism_plugins/`.
  Subdirectories are allowed and `__init__.py` files are optional.
- `load_plugins()` recursively visits those files in sorted path order and
  imports each one. The decorators run during import.
- `SimulationController` loads plugins during construction; the `polarism`
  CLI loads them before parsing flags. You can call `load_plugins(root)` to
  import from another project's root. It returns imported module names, or
  `[]` when the folder is absent.
- Python caches imported modules. Loading the same module a second time does
  not rerun its registration decorator. An import error propagates and stops
  setup; it is not silently skipped.
- A custom root is inserted into `sys.path`. Avoid reusing the same
  `polarism_plugins` module names from different roots in one process unless
  you deliberately manage Python's module cache.

## Registries and factories

| Component | Decorator import | Selection field | Factory call |
| --- | --- | --- | --- |
| [Laser](lasers.md) | `polarism.laser.register_laser` | `cfg.laser.laser_type` | `LaserClass(laser_config, X, Y, precision)` |
| [Potential](potentials.md) | `polarism.potential.register_potential` | `cfg.potential.potential_type` | `potential_fn(grid.X, grid.Y, potential_config)` |
| [Reservoir](reservoirs.md) | `polarism.reservoir.register_reservoir` | `cfg.reservoir.reservoir_type` | `ReservoirClass(reservoir_config, physics, grid, precision)` |
| [Solver](solvers.md) | `polarism.solver.register_solver` | `cfg.solver.method` | `SolverClass(config, grid)` |
| [Grid](grids.md) | `polarism.grid.register_grid` | `cfg.grid.grid_type` | `GridClass(grid_config)` |
| [Absorption](absorption.md) | `polarism.boundary_conditions.absorption.register_absorption` | `cfg.boundary_condition.absorption` | `StrategyClass(grid, boundary_config, physics)` |
| [Storage format](storage.md) | `polarism.results.storage.register_storage` | `cfg.result.formats` | `StorageClass(output_dir, batch_size, config, x, y)` |
| [Observable](observables.md) | `polarism.results.register_observable` | `cfg.result.observables` | `factory(controller)` |

In `multiple` laser mode, the factory calls the same laser constructor once per YAML entry. The other factories each create one selected object.

## Setup and step lifecycle

1. `SimulationController` configures NumPy or CuPy, loads plugins, validates
   the config, and checks known solver compatibility rules.
2. It creates the grid, boundary absorption strategy, potential, lasers,
   `PumpField`, reservoir, `SimulationState`, and solver, in that order.
3. It adds the strategy's `get_potential_distribution()` to the external
   potential. If outputs are enabled or a visitor is added, it builds result nodes; the reservoir's
   `make_result_nodes()` contributes optional nodes.
4. During a run, the controller calls `solver.advance(...)` for chunks between
   output samples. The inherited `advance()` calls `step(...)` repeatedly and
   updates `state.t` after each step.
5. A solver samples the callable pump at its stage times, advances the
   condensate and reservoir, and applies the boundary after-step action when
   its method requires it. The controller records results at configured times.

!!! note "Registration is separate from construction"
    Importing a plugin adds its class or function to a registry. The selected
    factory constructs it later, after the backend and grid are ready.

## Array and parameter rules

Use `compute_engine.xp` or an initialized `self.xp` so arrays stay on NumPy or CuPy as selected. `real_dtype(xp, precision)` and `complex_dtype(xp, precision)` in `polarism.config.dtype_utils` map `single` to float32/complex64 and `double` to float64/complex128. Fields use `(grid.ny, grid.nx)`; use that shape for pump, potential, reservoir, and wavefunction arrays. Avoid CPU transfers inside per-step methods on GPU runs.

`LaserParameters`, `GridParameters`, and `ReservoirParameters` have an `extra` dictionary. In multiple-laser YAML, unknown keys are merged into each laser's `extra`, alongside top-level `laser.extra` and an explicit entry-level `extra` mapping. These free-form keys are not individual CLI flags. Other configuration blocks do not have `extra`; use their existing fields or a component's own Python setup.

Registering a name already in a registry replaces the prior object and emits
`UserWarning`. Use distinct names unless you deliberately override a built-in.
Plugin files execute Python code on import, so run Polarism only from
directories whose plugins you trust.

## Try and test a plugin

The [example folder](https://github.com/jakrog01/Polarism/tree/master/examples/plugins) contains one working component of each type. From the repository root, this CPU check runs the examples in a few small simulations:

```bash
.venv/bin/pytest -q tests/integration/test_example_plugins.py
```

For your own project, write a pytest test using `tmp_path / "polarism_plugins"`, call `load_plugins(tmp_path)`, select your registered name in a small `Config`, run `SimulationController(cfg)`, and assert finite fields. Restore registry dictionaries and remove imported `polarism_plugins.*` modules after the test; see the example test for that cleanup. Then test on the intended GPU with `cfg.compute_engine.use_gpu = True` and compare a small case with the CPU result at a tolerance suited to your model. CUDA graph eligibility and solver limitations are described on each component page.

## Share a plugin

Copy or vendor the `polarism_plugins/` folder into the project where simulations run. Polarism does not discover installed packages through Python entry points. A Python package can still register components when imported explicitly: `import my_package` before creating `SimulationController`. Its import must execute the registration decorators. A later `load_plugins()` call for the current directory does not remove those registrations.
