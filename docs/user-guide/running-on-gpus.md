# Running on GPUs

Install a matching CuPy build and select a CUDA solver to advance a simulation on an NVIDIA GPU. Check the device and compare precision before scaling up.

## Select a device and solver

```yaml
solver:
  method: rk4-cuda
  precision: single
compute_engine:
  use_gpu: true
  gpu_device: 0
```

Install `polarism[cuda12]` for CUDA 12 or `polarism[cuda13]` for CUDA 13, with a compatible driver. `gpu_device` is a zero-based index among devices visible to the process. With `CUDA_VISIBLE_DEVICES=2,3`, index `1` selects physical GPU 3. If CuPy or a device is unavailable, the compute engine falls back to NumPy and CUDA-only solvers cannot run.

`rk4-cuda` supports periodic and closed-interval grids. `split-step-fft-cuda` and `ifrk-fft-cuda` require a periodic grid. `ifrk4-fft-cuda` is the fourth-order alias. See [Solver reference](../reference/solvers.md) for supported model features. `solver.precision` defaults to `double`; consumer GPUs can have much lower double-precision throughput. Compare accuracy and measured speed before choosing `single`.

## CUDA graphs and measurements

`solver.cuda_graphs` is enabled by default. Eligible solver chunks can be captured and replayed as CUDA graphs between scheduled samples. Eligibility depends on the laser, reservoir, absorption, and solver setup; capture failure falls back to ordinary launches. Set it to `false` to compare behavior or profile individual launches. Adaptive `ifrk-fft-cuda` uses the generic CuPy path and does not use graphs.

Use `polarism-bench --help` to inspect the benchmark CLI, then benchmark a representative grid, precision, reservoir, and method on your machine. Timing depends on device memory, kernel launch cost, and output cadence. For kernel and trace profiling, see [Profiling](../development/profiling.md).
