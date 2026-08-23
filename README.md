> "The test of a first-rate intelligence is the ability to hold two opposed ideas in the mind at the same time, and still retain the ability to function." ~ F. Scott Fitzgerald

# Polarism

High-performance tools for 2-D driven-dissipative Gross-Pitaevskii
simulations of exciton-polariton systems.

The repository is organized around one clear boundary:

- `polarism/` is the physics engine: model configuration, grids, lasers,
  reservoirs, solvers, state evolution, and result storage.
- `tests/` contains the validation layer for the engine and solver behavior.
- `src/` contains example applications and research workflows that use
  `polarism` as a library. These are not the core engine; they show how the
  engine is used for concrete studies, including the Slurm/Rysy
  `polariton_hpc_pipeline` pipeline.

The full documentation site is in `docs/` and is configured by `mkdocs.yml`.

Documentation site:
https://jakrog01.github.io/Polarism/#getting-started

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

src/                          Example applications built on `polarism`
  polariton_hpc_pipeline/      Slurm-oriented Rysy parameter-sweep pipeline
  dot_response_fit/           Dot-response analysis workflow
  threshold_finder/           Threshold-search workflow
  create_characteristic/      2-D pulse-energy/separation characteristic maps
  mnist_digits_polariton_snn_dynamic/
                               Dynamic MNIST SNN workflow with pitch/sigma sweep

docs/                         MkDocs Material documentation site
run.py                        Compatibility wrapper for the `polarism` CLI
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
.venv/bin/polarism --help
```

More examples are documented in
[`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md).

## Rysy / Slurm Pipeline

The active HPC campaign example lives in `src/polariton_hpc_pipeline/`.
Submit from a Rysy login node:

```bash
# Set POLARISM_ROOT to the Polarism checkout root on the cluster.
cd $POLARISM_ROOT/src/polariton_hpc_pipeline
bash submit.sh --config config.yaml --dry-run
bash submit.sh --config config.yaml
```

For local campaign YAMLs, use the ignored scenario directory:

```bash
bash submit.sh --config scenarios/<campaign>.yaml --dry-run
bash submit.sh --config scenarios/<campaign>.yaml
```

This workflow snapshots the config into a timestamped run directory under
`TETYDA_RUNS_BASE`, runs each scenario on NVMe scratch, renders plots or movies
inline, and copies lightweight artifacts back to persistent storage. The
pipeline entry point and resource model are documented in:

- [`src/polariton_hpc_pipeline/README.md`](src/polariton_hpc_pipeline/README.md)
- [`src/polariton_hpc_pipeline/cluster/README.md`](src/polariton_hpc_pipeline/cluster/README.md)
- [`src/polariton_hpc_pipeline/scenarios/README.md`](src/polariton_hpc_pipeline/scenarios/README.md)

## Dynamic MNIST SNN Workflow

The dynamic MNIST workflow lives in
[`src/mnist_digits_polariton_snn_dynamic/`](src/mnist_digits_polariton_snn_dynamic/).
It maps downsampled MNIST digits onto a 7x7 lattice of pulsed Gaussian pumps and
records dynamic readout traces for classification.

Local campaign diagnostics are intentionally ignored and are not part of the
repository contract. The tracked scenarios remain the reproducible inputs for
the workflow.

## Current Production Conventions

For pulsed-pump campaigns, prefer explicit physical semantics in YAML:

```yaml
global:
  laser_defaults:
    laser_type: pulse-gaussian
    power_definition: pulse_energy
```

`pulse_energy` means that `power` / `P` is the integrated dose of one pulse
over space and time. This keeps spot-size sweeps meaningful: increasing
`sigma_space` lowers the local peak pump while preserving the total pulse dose.
The legacy `peak_amplitude` mode is still available for reproducing old runs.

For artifact-sensitive campaigns, keep these diagnostics visible:

- `psi_sq.png` / `dynamics.mkv` for real-space condensate geometry.
- `psi_k.png` and scalar `high_k_frac_0p8_nyq` for high-k contamination.
- `P_max`, `P_area_integral`, and `P_cumulative_area_time_integral` for pump
  normalization checks.
- `scenario_meta.json` for effective power definition, grid spacing, solver
  method, initial-condition mode, and per-laser normalization metadata.

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
  for HDF5, scalar sidecars, plots, and movies.
- [`docs/examples/index.md`](docs/examples/index.md)
  for the documented Slurm workflows in `src/`.

## Development Notes

- Keep reusable physics logic inside `polarism/`.
- Keep tests for engine behavior inside `tests/`.
- Keep example-specific Slurm, scratch, manifest, and rendering orchestration inside
  the relevant `src/*/` pipeline and `cluster/` directories.
- Do not commit large run outputs. Persistent Rysy outputs belong under the run
  base configured in `slurm.env`; local campaign YAMLs belong in
  `src/polariton_hpc_pipeline/scenarios/`.
- Use `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/mkdocs` in this
  repository.

## Acknowledgements

This project is developed with the Exciton-Polariton research group at the
Faculty of Physics, University of Warsaw, with support from NCN OPUS grant work
and ICM computing infrastructure.
