# GPU solver profiling

Use a virtual environment with the project's development dependencies, `cupy-cuda13x[ctk]`, and `cupti-python`. Run on a machine with an NVIDIA GPU and a working driver. Nsight Systems (`nsys`) and Nsight Compute (`ncu`) come from the NVIDIA CUDA toolkit or Nsight packages, not pip.

## Repeatable benchmark

```bash
.venv/bin/python scripts/bench_gpu_solvers.py --label baseline
.venv/bin/python scripts/bench_gpu_solvers.py --label optimized
.venv/bin/python scripts/bench_gpu_solvers.py --compare artifacts/bench/baseline.json artifacts/bench/optimized.json
```

The benchmark writes JSON with the measurements and environment and a Markdown table under `artifacts/bench/`. `--compare` prints its table and writes `<before>_vs_<after>.md` in that directory. It uses a real controller, periodic square grids, a pulsed Gaussian pump, and disables all result outputs. The default run measures both GPU solvers on 256, 512, 1024, and 2048 grids, with all three reservoirs and both precisions. Each case is warmed up and timed repeatedly; automatic step count targets roughly half a second per repeat, with at least 20 steps. Use `--steps`, `--repeats`, and `--warmup` to set fixed durations, and `--cupti-steps` to control the separately traced interval. `--no-cupti` skips tracing.

`ms/step` is the median wall time per step across repeats, with device synchronization at each repeat boundary; JSON also holds each sample and the minimum. `Mpts·step/s` is grid points processed per second, in millions. `eff. GB/s` divides a **fixed reference model** of bytes per step by median step time, so future solver changes remain comparable. The RK4 model assumes cached stencil neighbors and the five-kernel structure of 0.2.x (rhs, three stages, combine); it is kept fixed as a reference even though rk4-cuda now launches four kernels per step. The FFT model counts only nine complex 2D transforms, one read and one write each, so its bandwidth is a lower bound. `% theor.` compares effective bandwidth with device memory clock × bus width × 2 ÷ 8 (or `--peak-gbps`); it says `n/a` if device attributes are unavailable. `% copy` compares with a measured large device-to-device copy, counting reads and writes.

`kernels/step`, `memcpy/step`, `memset/step`, and `syncs/step` come from CUPTI activities over a separate post-warmup interval. JSON breaks copies into HtoD, DtoH, and DtoD, synchronizations into stream, event, context, and stream-wait-event, and kernels into names with counts and microseconds per step. `allocs/step` counts CuPy pool requests during timed steps, including requests served from cached blocks. `GPU busy %` is the union of kernel execution intervals divided by the traced wall time; overlapping kernels count once. CUPTI columns show `n/a` if `cupti-python` is unavailable.

## Nsight Systems

`--profile` starts and stops the CUDA profiler around only the timed repeats and marks every step with an NVTX range. Capture that region with:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -o rk4 .venv/bin/python scripts/bench_gpu_solvers.py --solvers rk4-cuda --grids 1024 --reservoirs single --precisions double --repeats 1 --no-cupti --profile
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum rk4.nsys-rep
```

## Nsight Compute

Profile a few fused stages after launch warmup:

```bash
ncu --set full -k regex:fused_stage --launch-skip 40 --launch-count 3 -o rk4_stage .venv/bin/python scripts/bench_gpu_solvers.py --solvers rk4-cuda --grids 1024 --reservoirs single --precisions double --repeats 1 --no-cupti
```

Read `dram__throughput`, achieved occupancy, L1 texture cache hit rate (`l1tex`), and `sm__throughput` to distinguish memory traffic, occupancy, cache behavior, and compute throughput. A full NCU metric set slows execution substantially, so use the ordinary benchmark for timing.
