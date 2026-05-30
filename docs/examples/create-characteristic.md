# `create_characteristic` Pipeline

`src/create_characteristic/` builds two-dimensional threshold-characteristic
maps for pulsed polariton simulations.  It scans pulse energy and pulse
separation, runs one full GPE/reservoir simulation per grid point, and
aggregates `max(|psi|^2)` into heatmaps and threshold summaries.

The pipeline is intentionally scalar-only.  It is meant for broad preflight
maps where retaining full HDF5 trajectories for every point would be wasteful.

## Entry Point

Run from a Rysy login node:

```bash
bash src/create_characteristic/submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run] [--wait]
```

`CHARACTERISTIC_TIME` in `slurm.env` overrides the GPU walltime for this
pipeline.  If it is not set, the wrapper falls back to `SWEEP_TIME` and then
`SLURM_TIME`.

## Execution Model

```text
submit.sh
  -> validate config and slurm.env
  -> generate point_index.json from the 2D sweep grid
  -> GPU array over all but the last point
  -> GPU singleton for the last point
  -> CPU finalize after all points complete
```

As in `threshold_finder`, the simulation kernel in
`create_characteristic/simulation/core.py` is pure physics orchestration around
`polarism`.  Slurm submission, manifests, JSON output, and final plots are kept
in separate pipeline layers.

## Sweep Grid

The tracked default scans:

```yaml
sweep:
  energy_min: 1000.0
  energy_max: 3200.0
  energy_step: 100.0
  separation_min: 12.0
  separation_max: 122.0
  separation_step: 10.0
  post_pulse_time: 100.0
  adaptive_total_time: true
```

For each point, `pulse_energy` is assigned to `LaserParameters.P0/Pmax` and
`pulse_separation` is assigned to the pulsed Gaussian laser.  With finite pulse
trains and `adaptive_total_time: true`, the simulation time is:

```text
total_time = 2 * cutoff_sigma * sigma_time
           + (n_pulses - 1) * pulse_separation
           + post_pulse_time
```

This matters numerically: wide separation points need a longer physical window
to include the full pulse train and post-pulse relaxation.

## Configuration Notes

The default config uses the same GaAs/AlGaAs scale as the current scalar
preflights:

| Block | Default role |
| --- | --- |
| `global.grid` | `1024 x 1024`, `160 um x 160 um`, periodic grid |
| `global.physics` | `quadratic-double` GaAs working preset with filtered initial seed |
| `global.boundary_condition` | CAP with `sin2` profile |
| `global.solver` | `ifrk4-fft-cuda`, `dt: 0.01` |
| `laser` | single `pulse-gaussian`, `power_definition: pulse_energy` |
| `output.threshold_criterion` | condensation criterion for the binary threshold map |

`scalar_check_every` controls how often `max(|psi|^2)` is sampled during each
GPU point.  A smaller value catches narrow transient peaks more accurately but
increases scalar-reduction overhead.

## Outputs

Each run creates:

```text
cc_<timestamp>_<config-hash>/
  config.yaml
  manifest.json
  point_index.json
  points/
    point_000000.json
    point_000000_trace.npz   # optional
  characteristic_map.csv
  characteristic_map.json
  threshold_summary.json
  results/
    psi_max_heatmap.png
    psi_max_heatmap_log.png
    threshold_map.png
  logs/
```

`threshold_summary.json` stores the first pulse energy above
`output.threshold_criterion` for every pulse separation.  Diverged points are
kept in the map data and marked separately in the generated figures.

## When To Use It

Use this pipeline for coarse two-axis campaign design, for example choosing a
reasonable pulse-energy and separation region before launching heavier
multi-scenario runs.  If the physics question depends on field morphology,
animations, ROI traces, or fringe metrics, use `pump_multi_comparison` after
the scalar map has narrowed the parameter range.
