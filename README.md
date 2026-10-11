> "The test of a first-rate intelligence is the ability to hold two opposed ideas in the mind at the same time, and still retain the ability to function." ~ F. Scott Fitzgerald

# Polarism

Polarism is a Python package for 2-D driven-dissipative Gross-Pitaevskii simulations of exciton-polariton systems. It provides configurable pumps, reservoirs, solvers, and result output for CPU and optional CUDA runs.

[![PyPI version](https://img.shields.io/pypi/v/polarism.svg)](https://pypi.org/project/polarism/) [![Python versions](https://img.shields.io/pypi/pyversions/polarism.svg)](https://pypi.org/project/polarism/) [![Tests](https://github.com/jakrog01/Polarism/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/jakrog01/Polarism/actions/workflows/tests.yml) [![Documentation](https://img.shields.io/badge/docs-online-blue)](https://jakrog01.github.io/Polarism/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jakrog01/Polarism/blob/master/LICENSE)

## Highlights

- Finite-difference RK4, split-step FFT, ETD-RK2, and interaction-picture solver families.
- NumPy execution and optional CuPy GPU execution, including CUDA graph chunks for eligible `rk4-cuda` and `split-step-fft-cuda` runs.
- Single, double, and quadratic-double reservoirs; continuous and pulsed pumps.
- HDF5, JSON, and NPY output, live visualization, and online animation.
- Registries for custom solvers and model components, documented in the [plugin guide](https://jakrog01.github.io/Polarism/plugins/); a `tyro`-generated `polarism` CLI.
- `polarism-bench` for measuring per-step and chunked GPU execution on your hardware.

## Installation

```bash
pip install polarism
```

For CUDA runs, install the extra matching your CUDA environment:

| Extra | CUDA environment |
| --- | --- |
| `pip install "polarism[cuda12]"` | CUDA 12.x; use this for V100/Volta |
| `pip install "polarism[cuda13]"` | CUDA 13.x |

The installed CuPy build must be compatible with the NVIDIA driver and CUDA toolkit. See the [installation guide](https://jakrog01.github.io/Polarism/getting-started/installation/) and [GPU guide](https://jakrog01.github.io/Polarism/user-guide/running-on-gpus/).

For a development checkout:

```bash
pip install -e ".[dev]"
```

## Quickstart

This example matches [`examples/basic_run.py`](https://github.com/jakrog01/Polarism/blob/master/examples/basic_run.py):

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

For GPU execution, set `cfg.compute_engine.use_gpu = True`, choose `cfg.solver.method = "rk4-cuda"` or `"split-step-fft-cuda"`, and consider `cfg.solver.precision = "single"` on consumer GPUs. The FFT solver requires a periodic grid. Inspect CLI options with `polarism --help`.

## Choosing a solver

| Method | Grid | Device | Use |
| --- | --- | --- | --- |
| `rk4-fdm`, `rk4-fdm-fused` | periodic or closed-interval | CPU or CuPy | FDM reference or lower-overhead FDM |
| `rk4-cuda` | periodic or closed-interval | CUDA | Fused FDM runs |
| `split-step-fft`, `etd-rk2`, `ip-rk4` | periodic | CPU or CuPy | Spectral integration |
| `split-step-fft-cuda` | periodic | CUDA | Second-order spectral runs |
| `ifrk4-fft-cuda` | periodic | CUDA | Spectral RK4 with stage-coupled reservoirs |

Check timestep and grid convergence for your model. See the [solver reference](https://jakrog01.github.io/Polarism/api/solvers/) for compatibility details.

## Documentation

- [Getting started](https://jakrog01.github.io/Polarism/getting-started/)
- [User guide](https://jakrog01.github.io/Polarism/user-guide/grid-and-parameters/)
- [Running on GPUs](https://jakrog01.github.io/Polarism/user-guide/running-on-gpus/)
- [Plugin guide](https://jakrog01.github.io/Polarism/plugins/)
- [Reference](https://jakrog01.github.io/Polarism/api/overview/) and [Python API](https://jakrog01.github.io/Polarism/api/python-api/)
- [Physics background](https://jakrog01.github.io/Polarism/development/physics-background/)

For pulsed pumps, `power_definition: pulse_energy` makes `power` or `P` the integrated dose of one pulse over space and time. The legacy `peak_amplitude` mode remains available for reproducing older runs. See [Potentials and Pumps](https://jakrog01.github.io/Polarism/user-guide/potentials-and-pumps/).

## Contributing

See the [contributing guide](https://jakrog01.github.io/Polarism/development/contributing/). Run the default regression suite with `.venv/bin/pytest -q` from a development checkout.

## Citation

Use [CITATION.cff](https://github.com/jakrog01/Polarism/blob/master/CITATION.cff) or GitHub's “Cite this repository” menu.

## License

Polarism is licensed under the [MIT License](https://github.com/jakrog01/Polarism/blob/master/LICENSE).

## Acknowledgements

This project is developed with the Exciton-Polariton research group at the Faculty of Physics, University of Warsaw, with support from NCN OPUS grant work and ICM computing infrastructure.
