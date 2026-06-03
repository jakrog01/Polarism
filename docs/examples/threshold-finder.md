# `threshold_finder` Pipeline

`src/threshold_finder/` is a lightweight Rysy workflow for scalar response
sweeps.  It runs one full `polarism` GPE/reservoir simulation per sweep point
and records only the peak condensate density
`psi_sq_max = max(|psi|^2)`, divergence metadata, wall time, and optional scalar
traces.

Use it when the study question is a threshold or one-dimensional response
curve.  Use `polariton_hpc_pipeline` when the run needs multi-scenario field
plots, animations, ROI metrics, or cross-scenario visualization.

## Entry Point

Run from a Rysy login node:

```bash
bash src/threshold_finder/submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run] [--wait]
```

The wrapper validates `config.yaml` and `slurm.env`, creates a frozen run
directory, writes `power_index.json`, submits the GPU sweep, and submits a CPU
finalize job.  The project virtual environment is sourced automatically when
`.venv/bin/activate` or `venv/bin/activate` exists.

## Execution Model

```text
submit.sh
  -> GPU sweep array over all but the last point
  -> GPU singleton for the last point
  -> CPU finalize after all sweep jobs complete
```

The last point is submitted as a singleton instead of relying entirely on the
Slurm array.  This mirrors the current Rysy handling used by the project
wrappers and makes the final grid point explicit in the dependency chain.

The physics kernel lives in `threshold_finder/simulation/sweep_core.py`; it has
no Slurm or filesystem responsibility.  Config loading, validation, manifests,
and job submission stay in the config, manifest, stage, and shell-wrapper
layers.

## Sweep Modes

The default mode scans pump strength:

```yaml
sweep:
  P_min: 100.0
  P_max: 8000.0
  P_step: 100.0
```

The sweep variable can also be pulse separation:

```yaml
sweep:
  variable: pulse_separation
  P_fixed: 2200.0
  pulse_separation_min: 8.0
  pulse_separation_max: 80.0
  pulse_separation_step: 2.0
  adaptive_total_time: true
  post_pulse_time: 80.0
```

For finite pulse trains with `adaptive_total_time: true`, each point receives:

```text
total_time = 2 * cutoff_sigma * sigma_time
           + (n_pulses - 1) * pulse_separation
           + post_pulse_time
```

This avoids comparing long-separation points with too-short simulation windows.

## Configuration Notes

The tracked default is a GaAs/AlGaAs `quadratic-double` 9-pulse scalar scan
using `ifrk4-fft-cuda`, `pulse-gaussian`, `power_definition: pulse_energy`,
filtered stochastic initial conditions, and a CAP boundary.

Important fields:

| Section | Key | Meaning |
| --- | --- | --- |
| `global.grid` | `nx`, `ny`, `lx`, `ly` | spatial grid and physical box |
| `global.solver` | `method`, `dt`, `total_time` | solver backend and timestep budget |
| `global.physics` | GPE/reservoir constants | material and reservoir coefficients |
| `laser` | `sigma_space`, `sigma_time`, `pulse_separation`, `cutoff_sigma`, `n_pulses` | shared pulsed Gaussian shape |
| `laser` | `power_definition` | `peak_amplitude` or integrated `pulse_energy` |
| `sweep` | `scalar_check_every` | cadence for peak-density sampling |
| `sweep` | `early_stop_on_divergence` | stop a point immediately on NaN/Inf |
| `sweep` | `max_concurrent` | maximum concurrent GPU tasks |
| `output` | `save_per_power_trace` | write `power_<idx>_trace.npz` sidecars |

The validator includes a pump-growth guard for large local peak pump density
and a kinetic timestep warning for non-spectral explicit methods.  These checks
are preflight guards, not a proof of nonlinear stability; quantitative sweeps
still need a deliberate `dt` and grid-convergence check.

## Outputs

Each run is written under `TETYDA_RUNS_BASE` or `--runs-dir`:

```text
tf_<timestamp>_<config-hash>/
  config.yaml
  manifest.json
  power_index.json
  powers/
    power_000000.json
    power_000000_trace.npz   # optional
  threshold_curve.csv
  threshold_curve.json
  threshold_estimate.json
  results/
    psi_max_vs_power.png
    psi_max_vs_pulse_separation.png   # for separation sweeps
  logs/
```

`threshold_estimate.json` reports the first successful point with
`psi_sq_max >= 5e-2`.  For pulse-separation sweeps the threshold axis is
`pulse_separation`; for power sweeps it is `P`.

## Local Scenarios

Ad-hoc YAML files belong in `src/threshold_finder/scenarios/`.  Files matching
`src/threshold_finder/scenarios/*.yaml` are ignored by Git, so local production
campaigns can be staged there without changing the tracked default config.
