# Installation

Install the published package for simulation work. A development checkout is needed only when changing the source or building the docs.

## Install from PyPI

```bash
pip install polarism
```

Pin a release with `pip install "polarism==0.3.4"`.

The development version can be installed straight from GitHub:

```bash
pip install "git+https://github.com/jakrog01/Polarism.git@dev"
```

Check the installed CLI:

```bash
polarism --help
```

On a CUDA machine, `polarism-bench --help` shows the benchmark options; a small
benchmark can also confirm that the GPU backend is available. See
[Running on GPUs](../user-guide/running-on-gpus.md) for a command and guidance.
For a source checkout and test workflow, see [Contributing](../development/contributing.md).

## Optional GPU support

The runtime backend uses NumPy by default and switches to CuPy when `cfg.compute_engine.use_gpu = True` and CuPy is available. If CuPy is missing, the code falls back to CPU execution.

Recommended workflow:

- Start with a CPU install and validate correctness first.
- Add a CuPy build that matches the CUDA toolkit on the target machine:
  `pip install "polarism[cuda12]"` for CUDA 12.x or
  `pip install "polarism[cuda13]"` for CUDA 13.x.
- V100 (Volta) systems need `polarism[cuda12]`; CUDA 13 no longer supports Volta.
- Re-run the compliance or benchmark suites on GPU hardware before trusting production-scale timings.

For solver choice, precision, and CUDA graph chunks, see [Running on GPUs](../user-guide/running-on-gpus.md).
