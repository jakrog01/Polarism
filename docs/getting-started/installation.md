# Installation

Install Polarism with Python 3.12 or newer. The base package runs on a CPU; CUDA support is optional.

## Install the package

```bash
python -m pip install polarism
polarism --help
```

To pin the documented release, use `python -m pip install "polarism==0.9.0"`.

## Add CUDA support

Install the extra that matches your CUDA environment:

```bash
python -m pip install "polarism[cuda12]"
```

Use `polarism[cuda13]` for CUDA 13. Use CUDA 12 for V100 or Volta hardware. CuPy must match the NVIDIA driver and CUDA toolkit. Polarism compiles some kernels at runtime, so CUDA headers must be available. If a run reports `Failed to find CUDA headers`, load the toolkit or install the matching CuPy toolkit extra, such as `cupy-cuda12x[ctk]`.

Check whether CuPy can reach a CUDA device. The command prints `True` when one is available:

```bash
python -c "from polarism.compute_engine import has_cuda_device; print(has_cuda_device())"
```

Set `cfg.compute_engine.use_gpu = True` for a GPU run. See [Running on GPUs](../user-guide/running-on-gpus.md) for solver and precision choices.

## Install a development checkout

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

This installs the test and documentation dependencies. See [Contributing](../development/contributing.md) for the development workflow.
