# create_characteristic

`create_characteristic` measures threshold-crossing dynamics on a two-dimensional
parameter grid. It is the workflow for the question: how many times does the
global condensate density cross a chosen threshold as pulse energy and pulse
separation vary?

It is not a power-threshold calibration workflow; use `threshold_finder` for a
one-dimensional pump-power scan. It is not a general scenario and movie
pipeline; use `polariton_hpc_pipeline` for that.

## Outputs

For every `(pulse_energy, pulse_separation)` point, the pipeline runs a full
GPE-plus-reservoir simulation and writes:

- `results/psi_threshold_crossings_heatmap.png`: the number of both upward and
  downward crossings of `output.threshold_criterion` by `max|ψ|²`.
- `results/traces/point_<index>_trace.png`: the complete scalar time trace for
  that point, including threshold markers.
- `characteristic_map.csv`, `characteristic_map.json`, and
  `crossing_summary.json`: machine-readable result tables.

On every trace plot, a green `▲` marks an upward crossing and an orange `▼`
marks a downward crossing. A rise above the threshold followed by a fall below
it therefore contributes two crossings.

## Configuration

```yaml
global:
  grid:
    nx: 512
    ny: 512
    lx: 80.0
    ly: 80.0
  physics: {}
  boundary_condition: {}
  potential: {}
  reservoir: {}
  solver:
    dt: 0.01
    method: ifrk4-fft-cuda

laser:
  laser_type: pulse-gaussian
  sigma_space: 2.0
  sigma_time: 1.7
  cutoff_sigma: 3.0
  n_pulses: 9
  power_definition: pulse_energy

sweep:
  energy_min: 1500.0
  energy_max: 2500.0
  energy_step: 100.0
  separation_min: 20.0
  separation_max: 80.0
  separation_step: 10.0
  post_pulse_time: 80.0
  adaptive_total_time: true
  scalar_check_every: 100
  gain_check_every: 10
  max_concurrent: 8

output:
  save_per_point_trace: true
  threshold_criterion: 5.0e-2
```

Trace plots are sampled every `gain_check_every` simulation steps and always
include the initial and final samples. Set `save_per_point_trace: false` only
when the per-point trace plots are not required.

With `adaptive_total_time: true`, each point uses:

```text
total_time = 2 * cutoff_sigma * sigma_time
             + (n_pulses - 1) * pulse_separation
             + post_pulse_time
```

## Run on Rysy

```bash
bash src/create_characteristic/submit.sh --dry-run
bash src/create_characteristic/submit.sh
bash src/create_characteristic/submit.sh --config src/create_characteristic/scenarios/my_sweep.yaml
```

The submitter requires the repository-root `slurm.env` with
`SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_MEM`, `SLURM_GPUS`, `SLURM_CPUS`,
`SLURM_TIME`, `TETYDA_RUNS_BASE`, `FINALIZE_MEM`, `FINALIZE_CPUS`, and
`FINALIZE_TIME`.

## Run directory

```text
<run_dir>/
  config.yaml
  manifest.json
  point_index.json
  points/
    point_000000.json
  characteristic_map.csv
  characteristic_map.json
  crossing_summary.json
  results/
    psi_threshold_crossings_heatmap.png
    traces/
      point_000000_trace.png
  logs/
```

Local scenario files belong in [scenarios](scenarios/README.md).
