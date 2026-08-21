# Verification suite

## How to run each tier

- `pytest -m "not slow and not compliance"` runs the default unit and integration tiers.
- `pytest -m compliance` runs the reference and cross-check tiers.
- `pytest -m slow` runs convergence, memory, and GPU-speedup tiers.
- `pytest -m gpu` runs CPU/GPU agreement and speedup checks on CUDA hardware.
- Continuous integration runs `pytest -m 'not slow and not compliance and not gpu'` on `ubuntu-latest` for every push and PR.

The default command deliberately deselects `slow` and `compliance` nodes to keep ordinary development feedback short. A full validation pass must execute every tier above; the Phoenix reference matrix is run separately with `pytest tests/test_phoenix_benchmark.py -m slow --use-gpu` on CUDA hardware.

## Phoenix reference data

The eleven `test_phoenix_accuracy` comparisons cover three physical cases for
three FDM solvers plus two potential-free cases for `ifrk4-fft-cuda`. They read
their reference data from `tests/data/phoenix_benchmark/<case>/`. That directory
is tracked in the repository, so a fresh clone is enough to reproduce the
comparisons — there is no separate data download step. Each case stores
`rho_max.txt` (the `max|psi|^2` trace from PHOENIX), `psi_init.txt`, `pump.txt`,
`potential.txt`, `phoenix_lasers_setup.yaml`, `timing.json`, and
`frame_first.npz` with `frame_last.npz`. The test first checks that the Polarism
grid, pump, potential, and initial condition match the PHOENIX files, and only
then compares the result.

The `ifrk4-fft-cuda` thresholds describe an accepted limit of partial agreement,
not FDM-level agreement with PHOENIX. In case 01 the direct FDM-IFRK difference
of the density maximum at 500 ps was 0.05305, i.e. 97.1% of the final
IFRK-PHOENIX error of 0.05462. The L2 difference of the full density field did
not grow: it fell from 0.02412 at 10 ps to 0.01938 at 500 ps. The measurement
points to the spatial operator as the dominant source of the final scalar
discrepancy, but it does not confirm a monotonic divergence of the whole
trajectory or an influence of noise. Every new report records the interpretation
scope in the `validation_scope` field of `metrics.json` and in `metrics.txt`. The
`frame_phase_rmse` metric removes a constant global U(1) phase offset on the
reference density mask before comparing.

FDM directories under `tests/test_results/test_phoenix_benchmark/` without a
`metrics.json` predate the current report format. The zero value for unmeasured
memory use in their archived `metrics.txt` is not current measurement evidence.
Reports in the new format appear only after the corresponding case is re-run;
the repository does not treat the old image or text files as a result of the
present validation.

The data was generated with the `tests/data/phoenix_benchmark/example.ipynb`
notebook run inside the `robertschade/phoenix:latest` container (pyphoenix,
fp64, GPU). The files are marked binary in `.gitattributes` so the comparisons
stay byte-identical regardless of the `core.autocrlf` setting.

Without the `--use-gpu` flag pytest configures the CPU backend, including for
the Phoenix matrix. The flag is required to reproduce the `ifrk4-fft-cuda`
reports and the GPU results.

## Full CPU and GPU validation

Run the whole battery as a single process:

```bash
.venv/bin/pytest -q -m '' --use-gpu --tb=short \
  --junitxml=artifacts/reports/full_verification.xml
```

The command covers every marker, the full Phoenix matrix, and the CPU/GPU
agreement test. Do not run a second pytest process against the same GPU. Files
under `artifacts/` are run outputs and stay outside Git history.

## What each tier proves

Unit tests verify single-file correctness. Integration verifies the end-to-end run matrix. Reference tests compare against analytic closed forms. Convergence tests support order-of-accuracy claims. Cross-checks compare solvers and CPU/GPU parity. Quality tests cover memory scaling, extensibility, reproducibility, and GPU speedup.

## Artefacts

Generated outputs are `artifacts/convergence/*.json`, `artifacts/benchmark/gpu_speedup.json` with `environment` and `entries` keys, and `artifacts/benchmark/hardware.tex`.
Pipeline manifests and scenario metadata carry an `environment` object with the stable runtime fields; seeded threshold scans additionally write `threshold_ensemble.json`.

For the `rk4-fdm_single` time series the finest point `dt=0.000125` stays in the
regression: its error of `1.61e-6` is more than five orders of magnitude above
the `1e-11` floor, and the three final local slopes are 3.983, 3.986, and 3.973.
The lower `fit_r2=0.9968` is caused by the `dt=0.002` point with a local slope of
5.173, not by contamination of the finest point by the Richardson floor. The fit
still satisfies the order band and the quality criterion.

## Adding a solver

Register the solver in `polarism/solver/solver_registry.py` and add its `_SOLVER_CAPABILITIES` entry. The run matrix and convergence tiers derive their coverage from that capability table.
