# Running on GPUs

Install a CuPy extra before selecting a GPU solver. Use `pip install "polarism[cuda12]"` for CUDA 12.x, including V100/Volta systems, or `pip install "polarism[cuda13]"` for CUDA 13.x. The CuPy build must match the driver and CUDA environment. If CuPy or a device is unavailable, the compute engine reports a fallback to NumPy; CUDA-only solvers then cannot run.

## Choose a solver and precision

Set `cfg.compute_engine.use_gpu = True`. Use `rk4-cuda` for finite differences on periodic or closed-interval grids, `split-step-fft-cuda` for second-order spectral integration on a periodic grid, or `ifrk-fft-cuda` for selectable-order periodic spectral integration with stage-coupled reservoirs. `ifrk4-fft-cuda` remains an RK4 alias. See the [solver reference](../api/solvers.md) for reservoir and model-term support.

`rk4-cuda` performs each step in one tiled CUDA kernel for single precision,
and for double precision on GPUs classified as full-rate FP64. Other
double-precision GPUs use staged kernels. The tiled path also requires enough
shared memory for a tile. Tiled and staged results agree to rounding level.

`cfg.solver.precision` accepts `single` and `double` and defaults to `double`. The controller emits a one-time notice for double-precision runs on GPUs outside the high-throughput FP64 classes recognized by the code: compute capabilities 6.0 (P100), 7.0 (V100), 8.0 (A100), 9.0 (H100), and 10.0 (B200). Consumer GPUs often have reduced FP64 throughput. Compare accuracy and speed before switching to `single`.

## CUDA graph chunks

`cfg.solver.cuda_graphs` defaults to `True`. The controller groups solver steps between result samples, and `rk4-cuda`, `split-step-fft-cuda`, `ifrk-fft-cuda`, and its RK4 alias can replay eligible chunks as CUDA graphs. Graphs accept any number of lasers when each inherits `AbstractLaser.get_power` unchanged and has a cached spatial envelope, including qualifying plugin lasers. They also need an exact built-in reservoir type and absorption whose after-step action is a no-op (such as `no-absorption` or `cap`). The FFT solvers additionally require zero reservoir diffusion for their fused graph paths. A custom `get_power` uses regular solver launches. CUDA graph capture failure also falls back to regular launches. Set `cfg.solver.cuda_graphs = False` to disable graph replay.

## Measure your GPU

Run `polarism-bench --help` to inspect the CLI, then a small check such as:

```bash
polarism-bench --methods ifrk-fft-cuda --tableau rk4 --grid-sizes 128 --precisions single --steps 16
```

This command requires a working CUDA device. It prints device and CUDA metadata, measured device-copy bandwidth, per-step and chunked execution time, speedup, and throughput. Each case also prints a JSON record. These are measurements on your system, not promised performance. For deeper profiling, see [GPU Profiling](../development/profiling.md).

## Device selection

Set `cfg.compute_engine.gpu_device` to the index of the CUDA device to run on (default `0`). `compute_engine.configure()` makes that device current for the process, so every array, kernel and FFT plan created afterwards lives on it. An index outside the visible devices raises `ValueError`. Indices are counted among the devices visible to the process: with `CUDA_VISIBLE_DEVICES=2,3`, `gpu_device = 1` selects physical GPU 3. On multi-GPU nodes, run one simulation per GPU, either by setting `gpu_device` per run or by giving each process its own `CUDA_VISIBLE_DEVICES`.
