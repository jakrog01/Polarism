# Plugins

You can add your own lasers, potentials, solvers, reservoirs, grids and
absorption strategies
without modifying Polarism. Create a `polarism_plugins/` folder in your project
(the directory you run `polarism` or your script from) and register components
with the usual decorators.

## Folder layout

```
my_project/
  polarism_plugins/
    laser/
      ring_laser.py
    potential/
      my_potential.py
  lasers_setup.yaml
```

Every `.py` file under `polarism_plugins/` is imported when a
`SimulationController` is created or the `polarism` CLI starts. Subfolders are
only for organisation; `__init__.py` files are optional. The registry a
component lands in is decided by its decorator:

| Component  | Decorator             | Import from                              |
|------------|-----------------------|------------------------------------------|
| Laser      | `register_laser`      | `polarism.laser`                         |
| Potential  | `register_potential`  | `polarism.potential`                     |
| Solver     | `register_solver`     | `polarism.solver`                        |
| Absorption | `register_absorption` | `polarism.boundary_conditions.absorption` |
| Reservoir  | `register_reservoir`  | `polarism.reservoir`                     |
| Grid       | `register_grid`       | `polarism.grid`                          |

## Example: custom laser

```python
# polarism_plugins/laser/ring_laser.py
from polarism.laser import register_laser
from polarism.laser.abstract_laser import AbstractLaser


@register_laser("ring")
class RingLaser(AbstractLaser):
    def __init__(self, laser_config, X, Y, precision="double"):
        super().__init__(laser_config, X, Y, precision)
        self.sigma_space = laser_config.sigma_space
        self.radius = laser_config.extra.get("ring_radius", 5.0)
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        r = self.xp.sqrt((X - self.x0) ** 2 + (Y - self.y0) ** 2)
        return self.xp.exp(-0.5 * (r - self.radius) ** 2 / self.sigma_space**2)

    def _P_time(self, t):
        return 1.0
```

Then select it like any built-in laser:

```python
cfg.laser.laser_type = "ring"
```

## Solvers

Plugin solvers implement `step(potential, pump, reservoir, boundary_condition,
state)`. The `pump` argument is now a callable `pump(t) -> array`, replacing the
former array argument. Use `state.t` as the step start time and evaluate the
pump at the integration stage times. The caller updates `state.t` after the
step. See the [solver interface](../api/solvers.md#solver-interface) for the
built-in stage schedules.

## Reservoirs

Subclass `AbstractReservoir` (or a built-in reservoir) and register it:

```python
from polarism.reservoir import register_reservoir
from polarism.reservoir.single_reservoir import SingleReservoir


@register_reservoir("my-reservoir")
class MyReservoir(SingleReservoir):
    ...
```

Plugin reservoirs work with every solver that drives the reservoir through the
`AbstractReservoir` interface. `rk4-cuda` has hard-coded reservoir kernels and
rejects them; `split-step-fft` accepts them with second-order Strang
coupling between the condensate and reservoir.

## Grids

Solvers choose their boundary handling from the grid's `boundary` class
attribute, which must be `"periodic"` or `"closed-interval"`. The simplest way
is to subclass a built-in grid:

```python
from polarism.grid import register_grid
from polarism.grid.periodic_simulation_grid_2d import PeriodicSimulationGrid2D


@register_grid("my-grid")
class MyGrid(PeriodicSimulationGrid2D):
    ...
```

Solver compatibility is checked against `boundary`, so `my-grid` above is
accepted by every solver that supports periodic grids.

## Custom parameters

`LaserParameters.extra` is a free-form dictionary for plugin-specific settings.
In `multiple` laser mode, any YAML key that is not a standard laser field is
placed in `extra`:

```yaml
lasers:
  - laser_type: ring
    ring_radius: 8.0
    P0: 0.5
```

From Python, set it directly: `cfg.laser.extra = {"ring_radius": 8.0}`.
`GridParameters` and `ReservoirParameters` have the same `extra` field for grid
and reservoir plugins. `extra` is not exposed as a CLI flag.

## Notes

- Registering a name that already exists (including a built-in one) overrides
  it and emits a `UserWarning`.
- Plugins are regular Python code executed from the current working directory.
  Only run Polarism in directories whose `polarism_plugins/` you trust.
- To load plugins from another location, call `polarism.load_plugins(root)`
  before creating the `SimulationController`.
