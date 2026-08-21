# Quickstart

The fastest path to a first run is a small 2-D simulation driven through the public Python API.

## Minimal example

```python
import polarism as ps
from polarism.compute_engine import compute_engine

cfg = ps.Config()

cfg.grid.nx = 128
cfg.grid.ny = 128
cfg.grid.lx = 100.0
cfg.grid.ly = 100.0
cfg.grid.grid_type = "periodic"

cfg.reservoir.reservoir_type = "single"
cfg.potential.potential_type = "zero"

cfg.laser.mode = "single"
cfg.laser.laser_type = "continuous-gaussian"
cfg.laser.P0 = 0.02
cfg.laser.sigma_space = 12.0

cfg.solver.method = "rk4-fdm"
cfg.solver.dt = 1e-3
cfg.solver.total_time = 2.0

cfg.result.save_results = True
cfg.result.save_hdf5 = True
cfg.result.save_interval = 50
cfg.result.output_directory = "quickstart_results"

cfg.compute_engine.use_gpu = False
compute_engine.configure(cfg.compute_engine)

controller = ps.SimulationController(cfg)
controller.run()
```

Save the snippet to `quickstart.py` and run:

```bash
python quickstart.py
```

If you prefer not to write a Python driver script, the installed `polarism`
command uses `tyro` to expose `ps.Config` as command-line flags:

```bash
polarism \
  --grid.nx 128 \
  --grid.ny 128 \
  --reservoir.reservoir-type single \
  --potential.potential-type zero \
  --laser.mode single \
  --laser.laser-type continuous-gaussian \
  --laser.P0 0.02 \
  --solver.method rk4-fdm \
  --solver.total-time 2.0 \
  --result.save-results \
  --result.save-hdf5
```

Use `polarism --help` to inspect the full set of generated options. The
repository-root `run.py` remains as a compatibility wrapper for this command.

## What this example does

- Builds a periodic 2-D grid.
- Uses the single-reservoir model.
- Applies a Gaussian continuous pump and zero external potential.
- Evolves the system with the reference `rk4-fdm` solver.
- Writes batched HDF5 output into `quickstart_results/`.

## Suggested next steps

- Switch `cfg.solver.method` to `rk4-fdm-fused` or `rk4-cuda` after you have a stable reference run.
- Enable `cfg.result.real_time_view = True` for interactive inspection during local runs.
- Move to [Grid and Parameters](../user-guide/grid-and-parameters.md) when you need to tune the physical model.

## Notes on accuracy

There is no built-in imaginary-time driver in the current codebase. For production runs, treat `dt`, grid spacing, and solver choice as convergence parameters and validate them with the test strategy described in [Testing and CI](../development/testing-and-ci.md).
