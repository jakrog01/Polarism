# Polarism

High-performance tools for 2-D driven-dissipative Gross-Pitaevskii
simulations of exciton-polariton systems.

The repository is organized around one clear boundary:

- `polarism/` is the physics engine: model configuration, grids, lasers,
  reservoirs, solvers, state evolution, and result storage.
- `tests/` contains the validation layer for the engine and solver behavior.

The full documentation site is in `docs/` and is configured by `mkdocs.yml`.

Documentation site:
https://jakrog01.github.io/polarism/#getting-started

## Repository Layout

```text
polarism/                     Physics engine and reusable Python package
  config/                     Dataclasses and validation helpers
  grid/                       Periodic and closed-interval 2-D grids
  laser/                      Continuous and pulsed pump definitions
  reservoir/                  Single, double, and quadratic-double reservoirs
  solver/                     CPU, CuPy, CUDA RK4, FFT, and ETD/IP solvers
  results/                    Storage and visualization support

tests/                        CPU, compliance, and benchmark-style tests

docs/                         MkDocs Material documentation site
run.py                        Tyro CLI over the public `polarism.Config`
pyproject.toml                Project metadata and dependency groups
```

## Quick Local Setup

Use the project virtual environment. The global Python on the target systems is
not expected to contain the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run the default CPU-oriented checks:

```bash
.venv/bin/pytest
.venv/bin/mkdocs build
```

The default pytest selection excludes slow and compliance-marked suites. See
[`docs/development/testing-and-ci.md`](docs/development/testing-and-ci.md) for
the heavier validation commands.

For the complete CPU/GPU validation, including the Phoenix reference matrix:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

Run this command as a single process with exclusive access to the GPU.

## Minimal API Example

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
cfg.result.output_directory = "quickstart_results"

cfg.compute_engine.use_gpu = False
compute_engine.configure(cfg.compute_engine)

ps.SimulationController(cfg).run()
```

For the command-line interface generated from `polarism.Config`:

```bash
.venv/bin/python run.py --help
```

More examples are documented in
[`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md).

## Pump Power Conventions

For pulsed pumps, prefer explicit physical semantics in the laser YAML:

```yaml
lasers:
  - laser_type: pulse-gaussian
    power_definition: pulse_energy
```

`pulse_energy` means that `power` / `P` is the integrated dose of one pulse
over space and time. This keeps spot-size sweeps meaningful: increasing
`sigma_space` lowers the local peak pump while preserving the total pulse dose.
The legacy `peak_amplitude` mode is still available for reproducing old runs.

Numerical choices such as grid spacing, timestep, boundary condition, kinetic
relaxation, and Laplacian stencil should be treated as convergence parameters,
not as cosmetic post-processing settings. The model background and rationale are
summarized in
[`docs/development/physics-background.md`](docs/development/physics-background.md).

## Documentation

Build and serve the documentation locally:

```bash
.venv/bin/mkdocs serve
```

Key entry points:

- [`docs/index.md`](docs/index.md) for the documentation map.
- [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
  for setup.
- [`docs/user-guide/grid-and-parameters.md`](docs/user-guide/grid-and-parameters.md)
  for spatial discretization and physical parameters.
- [`docs/user-guide/potentials-and-pumps.md`](docs/user-guide/potentials-and-pumps.md)
  for laser and pump definitions.
- [`docs/user-guide/output-and-visualization.md`](docs/user-guide/output-and-visualization.md)
  for HDF5 output, plots, and movies.

## Development Notes

- Keep reusable physics logic inside `polarism/`.
- Keep tests for engine behavior inside `tests/`.
- Use `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/mkdocs` in this
  repository.

## Acknowledgements

This project is developed with the Exciton-Polariton research group at the
Faculty of Physics, University of Warsaw, with support from NCN OPUS grant work
and ICM computing infrastructure.
