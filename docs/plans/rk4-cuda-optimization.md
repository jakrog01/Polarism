# Plan: rk4-cuda optimization for 0.3.0

Baseline: `artifacts/bench/baseline.{json,md}` (RTX 4070 SUPER, CUDA 13.2,
CuPy 14.2), Phoenix wall times in `artifacts/bench/phoenix_baseline_metrics/`
(rk4-cuda: 38 s / 76 s / 76 s).

## What the baseline shows

Per-step GPU time, rk4-cuda, 2048², double precision, single reservoir
(7.91 ms/step, 52 % of theoretical bandwidth):

| part | us/step | note |
| --- | ---: | --- |
| 3 x fused_stage_rhs | 3124 | ~80 % of copy bandwidth each |
| rk4_combine | 1449 | reads psi0 + 4 k, pure traffic |
| gpe_rhs | 752 | |
| pump: 3 x (zeros + 2 mul + add) | ~1680 | Python-side laser sum, 3 evaluations |
| 2 x maximum | 331 | duplicate clamp (kernel already clamps) |
| 3 x DtoD memcpy + copy kernels | ~600 | psi_buf / psi0 / nR0 copies |

At 256² the GPU is busy 86 us of 220 us per step: 16-18 kernel launches,
3 memsets, 3 copies, 2 synchronizations and 11-13 pool allocations per step
dominate. Kernels in single precision contain no f64 instructions (checked
in PTX for all 36 variants).

## Steps (one commit each; order = gain / risk)

### P0 + P1 are done together (one commit)

Measured race: two identical rk4-cuda runs (256², 3200 steps) differ by
4e-5 .. 9e-4 relative L2; with the solver's streams replaced by the null
stream the result is bit-reproducible and the time error converges with
order 4 (4.1e-8, 5.0e-9, 3.2e-10, 2.0e-11 for dt = 0.032 .. 0.004). The new
guard tests would fail on the current code, so they land with the P1 fix.
Add a determinism test: two identical GPU runs must be bitwise equal.

### P0. Guard tests (no solver change)

Today the CUDA kernels of rk4-cuda are covered only by the Phoenix
benchmark: `test_cpu_gpu_agreement` runs the default `rk4-fdm`,
`test_pump_time_order` and compliance run rk4-cuda's CPU fallback, and
`test_time_convergence` does not include rk4-cuda.

- `tests/crosscheck/test_rk4_cuda_agreement.py` (`gpu` marker): rk4-cuda on
  GPU vs rk4-fdm (same config) for reservoir {single, double,
  quadratic-double} x grid {periodic, closed-interval} x precision
  {double: rel L2 < 1e-10, single: < 1e-5}, with a time-dependent pulsed
  pump, two lasers, a CAP absorber (complex potential), nonzero
  `kinetic_relaxation_eta` and reservoir diffusion, ~50 steps. The
  `isotropic-9pt` Laplacian (only rk4-cuda has it) is checked against
  rk4-cuda's CPU path.
- `test_time_convergence`: add rk4-cuda on GPU, expected order 4.

### P1. Step driver: streams, copies, aliasing, potential cache (issues 1, 2, 4)

- Remove `_stream_compute`, `_stream_transfer`, the event and the per-step
  `synchronize()`; launch everything on the current CuPy stream, so the
  pump, potential and reservoir arrays produced on that stream are ordered
  correctly and steps stay asynchronous.
- Read `state.psi` / reservoir arrays directly as kernel inputs (flat
  contiguous views, cast only if dtype differs); write the combine output
  into freshly pool-allocated arrays and hand those to `state.psi` /
  `reservoir.set_state` (no `psi_buf`/`psi0` copies, no aliasing of internal
  buffers: a reference to `state.psi` kept by a caller is never overwritten).
- Drop the solver-side `xp.maximum` (kernel clamps; `set_state` clamps
  anyway).
- Potential cache: compare the potential object with `is` and keep a
  reference, instead of `id()`.
- Unify the three `_step_gpu*` / `_gpu_*` variants into one driver that
  builds the argument lists per reservoir type (only because P1 rewrites
  `step()` anyway and P3 would otherwise triple the work).
- New test: holding `psi_before = state.psi` across a step leaves it
  unchanged; GPU state dtypes stay float32/complex64 in single precision.

Expected: -3 memcpy, -2 maximum, -2 syncs per step; ~0.9 ms at 2048² double
and most of the small-grid overhead.

### P2. Pump evaluation (issue 3, generic part)

- `AbstractLaser.get_power`: compute the scalar `amplitude * P_time` first,
  then one multiply with the cached envelope (1 kernel instead of 2); return
  a cached zero array before `delay` instead of allocating.
- `SimulationController._compute_total_pump`: single laser -> return its
  field directly; several lasers -> accumulate in place, no `zeros` + add.
- Benefits every solver. Rounding of the pump changes at the 1-ulp level;
  checked by crosscheck + Phoenix.
- In-kernel pump fusion (pass envelopes and per-stage scalars to the stage
  kernel) is deferred: it needs a new pump interface between controller and
  solver and must keep non-separable plugin lasers working. Decide after P2
  measurements.

### P3. Fuse the RK4 combine into stage 4 (issue 5)

Stage 4 computes k4 at its own point only, so it can write
`psi0 + dt/6 (k1 + 2 k2 + 2 k3 + k4)` (and the clamped reservoir) directly:
one kernel less per step and ~96 B/pt less traffic in double precision
(expected ~1 ms at 2048² double). Results change only by floating-point
reassociation, if at all.

### P4. Kernel tuning experiments (issue 5) — keep only what measures faster

- compile-time flag to skip the `V_im` load when the potential is real;
- 2D launch grid (no `tid / nx`, `tid % nx`) and block size chosen by timing
  instead of occupancy;
- shared-memory tiling of the stencil: the stage kernels already reach
  ~80 % of copy bandwidth and neighbours hit the 48 MB L2, so the expected
  gain is small; try once, keep only with a clear win.

### P5. CPU fallback (issue 6) — needs a decision

Options: (a) remove it and raise when no GPU (behaviour change; tests that
run rk4-cuda on CPU move to rk4-fdm-fused); (b) delegate to rk4-fdm-fused
and drop ~280 lines, losing `isotropic-9pt` on CPU (it exists only in
rk4-cuda); (c) keep one generic CPU path built on
`reservoir.get_derivatives` (like rk4-fdm), keeping 9pt, removing the three
hard-coded reservoir variants. Recommendation: (c).

### Out of scope here, reported separately

- `ifrk4-fft-cuda` promotes single precision to complex128 (propagators are
  built from float64 `k_squared`) and fails for quadratic-double in single.
- `ifrk4-fft-cuda` runs 150-180 unfused elementwise kernels per step
  (45-54 ms at 2048²); relevant to the split-step decision (step 3).

## Gate for every step

- `.venv/bin/python -m pytest -q -p no:cacheprovider`
- `.venv/bin/python -m pytest -q -m '' --use-gpu -p no:cacheprovider tests/crosscheck tests/convergence`
- `.venv/bin/python -m pytest -q -m '' --use-gpu -p no:cacheprovider tests/test_phoenix_benchmark.py -k rk4-cuda`
- `.venv/bin/python scripts/bench_gpu_solvers.py --solvers rk4-cuda --label <step>`
  and `--compare artifacts/bench/<previous>.json artifacts/bench/<step>.json`
- Full `pytest -m '' --use-gpu` once before the release.
