> "The test of a first-rate intelligence is the ability to hold two opposed ideas in the mind at the same time, and still retain the ability to function." ~ F. Scott Fitzgerald

# Polarism

Polarism is a Python package for 2-D driven-dissipative Gross-Pitaevskii simulations of exciton-polariton systems. It provides configurable pumps, reservoirs, solvers, and result output for CPU and optional CUDA runs.

[![PyPI version](https://img.shields.io/pypi/v/polarism.svg)](https://pypi.org/project/polarism/) [![Python versions](https://img.shields.io/pypi/pyversions/polarism.svg)](https://pypi.org/project/polarism/) [![Tests](https://github.com/jakrog01/Polarism/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/jakrog01/Polarism/actions/workflows/tests.yml) [![Documentation](https://img.shields.io/badge/docs-online-blue)](https://jakrog01.github.io/Polarism/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/jakrog01/Polarism/blob/master/LICENSE) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakrog01/Polarism/blob/master/examples/colab_quickstart.ipynb) [![DOI](https://zenodo.org/badge/1412443657.svg)](https://doi.org/10.5281/zenodo.23531116)

<p align="center">
  <img src="https://raw.githubusercontent.com/jakrog01/Polarism/master/docs/assets/readme/ring_vortices.webp" alt="Condensate density and phase under a ring-shaped pump, with vortices forming and moving inside the ring" width="608">
</p>
<p align="center"><sub>A ring-shaped pump (the <a href="#adding-your-own-pump">example plugin below</a>) traps a rotating condensate with quantized vortices. Left: density |ψ|², right: phase. 256² grid; 180 frames spanning 300–1200 ps, rendered by <a href="https://github.com/jakrog01/Polarism/blob/master/examples/readme_media.py"><code>examples/readme_media.py</code></a>.</sub></p>

## Highlights

- Finite-difference RK4, split-step FFT, ETD-RK2, and interaction-picture solver families.
- NumPy execution and optional CuPy GPU execution, including CUDA graph chunks for eligible CUDA solver runs.
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

Open a saved run with `ps.load_results(run_dir)`. Its `fields["density"]`,
`scalars["norm"]`, and `static["pump"]` use stable ASCII names; `x` and `y`
contain the grid coordinates. HDF5 datasets are read lazily, so `fields["density"][::10]` loads every tenth frame only. Use the returned object as a context manager for HDF5 output.

For GPU execution, set `cfg.compute_engine.use_gpu = True`, choose `cfg.solver.method = "rk4-cuda"`, `"split-step-fft-cuda"`, or `"ifrk-fft-cuda"`, and consider `cfg.solver.precision = "single"` on consumer GPUs. FFT solvers require a periodic grid. Inspect CLI options with `polarism --help`.

To try Polarism on a GPU without installing CUDA, open the [Colab notebook](https://colab.research.google.com/github/jakrog01/Polarism/blob/master/examples/colab_quickstart.ipynb).

## Adding your own pump

Pumps, reservoirs, potentials, solvers, grids, and absorption strategies are selected by name from registries. Subclass the base class, register it, and select the name in the config. This ring-shaped pump traps vortices inside the ring:

```python
import polarism as ps
from polarism.laser import register_laser
from polarism.laser.abstract_laser import AbstractLaser


@register_laser("ring")
class RingLaser(AbstractLaser):
    """Continuous Gaussian ring of radius `ring_radius` and width `sigma_space`."""

    def __init__(self, laser_config, X, Y, precision="double"):
        super().__init__(laser_config, X, Y, precision)
        self.radius = laser_config.extra.get("ring_radius", 20.0)
        self.width = laser_config.sigma_space
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        r = self.xp.sqrt((X - self.x0) ** 2 + (Y - self.y0) ** 2)
        return self.xp.exp(-0.5 * ((r - self.radius) / self.width) ** 2)

    def _P_time(self, t):
        return 1.0


cfg = ps.Config()
cfg.laser.laser_type = "ring"
cfg.laser.extra = {"ring_radius": 20.0}
cfg.laser.sigma_space = 4.0
```

The class works on both NumPy and CuPy through `self.xp`. Put it in `polarism_plugins/` to have it discovered automatically, including by the `polarism` CLI. The [plugin guide](https://jakrog01.github.io/Polarism/plugins/) covers the contract for each component type.

## Choosing a solver

| Method | Grid | Device | Use |
| --- | --- | --- | --- |
| `rk4-fdm`, `rk4-fdm-fused` | periodic or closed-interval | CPU or CuPy | FDM reference or lower-overhead FDM |
| `rk4-cuda` | periodic or closed-interval | CUDA | Fused FDM runs |
| `split-step-fft`, `etd-rk2`, `ip-rk4` | periodic | CPU or CuPy | Spectral integration |
| `split-step-fft-cuda` | periodic | CUDA | Second-order spectral runs |
| `ifrk-fft-cuda` | periodic | CPU or CUDA | Spectral RK4/DP5/DOP853/Verner 7/9 with stage-coupled reservoirs |
| `ifrk4-fft-cuda` | periodic | CPU or CUDA | Historical alias for spectral RK4 |

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

Cite Polarism with its Zenodo DOI, [10.5281/zenodo.23531116](https://doi.org/10.5281/zenodo.23531116), which always resolves to the latest release; each release also has its own version DOI on Zenodo. The same metadata is in [CITATION.cff](https://github.com/jakrog01/Polarism/blob/master/CITATION.cff) and GitHub's “Cite this repository” menu.

## License

Polarism is licensed under the [MIT License](https://github.com/jakrog01/Polarism/blob/master/LICENSE).

## Acknowledgements

This project is developed with the Exciton-Polariton research group at the Faculty of Physics, University of Warsaw, with support from NCN OPUS grant work and ICM computing infrastructure.
