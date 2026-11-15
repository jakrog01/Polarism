# Configuring a run

Start with `ps.Config()` and change only the fields your run needs. The same nested field names work in Python, YAML and the `polarism` CLI.

```python
import polarism as ps

cfg = ps.Config()
cfg.grid.nx = 64
cfg.grid.ny = 64
cfg.grid.lx = 60.0
cfg.grid.ly = 60.0
cfg.laser.laser_type = "continuous-gaussian"
cfg.laser.P0 = 0.3
cfg.laser.sigma_space = 6.0
cfg.solver.method = "ifrk-fft-cuda"
cfg.solver.tableau = "dop853"
cfg.solver.adaptive = True
cfg.solver.dt = 0.1
cfg.solver.total_time = 150.0
ps.SimulationController(cfg).run()
```

This runs on the CPU for 150 ps in a 60 µm by 60 µm box. The [quickstart](../getting-started/quickstart.md) shows how to save results.

## Configuration blocks

| Block | Sets |
| --- | --- |
| `grid` | Dimension, box size and grid topology |
| `physics` | Mass, interactions, rates, dispersion and spin |
| `laser` | Incoherent pump |
| `reservoir` | Reservoir model and component count |
| `potential` | External potential and its time envelope |
| `drive` | Coherent source and rotating frame |
| `initial_condition`, `noise` | Starting field and random increments |
| `boundary_condition` | Edge absorption |
| `solver`, `compute_engine` | Time integration and CPU or GPU |
| `result` | Fields, observables, visualization and storage |
| `checkpoint`, `stop`, `sweep` | Restart, early stopping and sweep controls |
| `terms` | Additional registered model terms |

See [Configuration reference](../reference/configuration.md) for every key, default and allowed value.

## Python, YAML and CLI

Save a Python config with `cfg.to_yaml("run.yaml")`. Load and validate it with `ps.Config.from_yaml("run.yaml")`. A YAML file can contain only the blocks you want to change:

```yaml
grid:
  nx: 64
  ny: 64
  lx: 60.0
  ly: 60.0
laser:
  laser_type: continuous-gaussian
  P0: 0.3
  sigma_space: 6.0
solver:
  method: ifrk-fft-cuda
  tableau: dop853
  adaptive: true
  dt: 0.1
  total_time: 150.0
```

Run it with `polarism --config run.yaml`. CLI flags override file values, for example `polarism --config run.yaml --grid.nx 32`. Use `polarism --help` to see the generated flags.

Field names are exact and case sensitive: `laser.P0` is valid; `laser.p0` is not. Assignment in Python rejects unknown fields. `Config.from_yaml` also rejects unknown blocks and fields, often with a suggested spelling. Values are checked when YAML loads and when `SimulationController` starts; a direct Python assignment alone does not validate a value.

## Units

Enter all lengths in µm, times in ps and energies in meV. Coordinates and profile widths are in µm; decay rates are in ps⁻¹. In two dimensions, density is in µm⁻² and a continuous pump `P` is in µm⁻² ps⁻¹. The code does no unit conversion. See [Grid and physics](grid-and-physics.md) for the model parameters.
