# Profiling

Measure a representative simulation before changing a solver for speed. Use the bundled benchmark for comparable run timings, then inspect traces on an NVIDIA GPU if a bottleneck remains unclear.

## Repeatable benchmark

```bash
.venv/bin/python scripts/bench_gpu_solvers.py --label baseline
.venv/bin/python scripts/bench_gpu_solvers.py --label candidate
.venv/bin/python scripts/bench_gpu_solvers.py --compare artifacts/bench/baseline.json artifacts/bench/candidate.json
```

Install the development extra, a matching CuPy package, and CUDA toolkit headers before running GPU cases. The script writes JSON and a Markdown table under `artifacts/bench/`. It warms each case and reports median wall time per step after device synchronization. Its effective bandwidth is based on a fixed reference traffic model, so it is a comparison metric rather than measured DRAM bytes. Use `--solvers`, `--grids`, `--reservoirs`, `--precisions`, `--steps`, `--warmup`, and `--repeats` to narrow or control runs. `--no-cupti` skips optional CUPTI tracing; install `cupti-python` when kernel and synchronization counts are needed.

For a short device-specific check, `polarism-bench --help` lists the installed CLI options. Compare the same grid, precision, reservoir, and output schedule when measuring changes. Avoid another GPU workload during timing.

## Trace CUDA work

Nsight Systems can show launch gaps, transfers, and synchronization. Use `--profile` with the benchmark to mark the timed interval, then capture its CUDA profiler API range with `nsys`. Nsight Compute can inspect a selected kernel's achieved occupancy and memory throughput with `ncu`. Full metric collection slows execution; use the ordinary benchmark for elapsed-time comparisons. Neither Nsight tool is installed by the Python package.
