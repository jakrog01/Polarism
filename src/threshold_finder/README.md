# threshold_finder

Sweeps laser pump power P from `P_min` to `P_max` in steps of `P_step`, runs one full GPE+reservoir simulation per power point, and collects `psi_sq_max = max(|ψ|²)` for each P.

## Pipeline stages

```
[login node]  validate config + generate power_index.json
     |
[GPU array]   one Slurm task per power value  → powers/power_<idx>.json
     |
[CPU job]     aggregate results               → threshold_curve.csv / .json
                                              → results/psi_max_vs_power.png
```

## Quick start (Rysy login node)

```bash
# dry run — prints planned jobs without submitting
bash src/threshold_finder/submit.sh --dry-run

# real submit with default config
bash src/threshold_finder/submit.sh

# custom config
bash src/threshold_finder/submit.sh --config /path/to/my_config.yaml

# block until all jobs finish
bash src/threshold_finder/submit.sh --wait
```

## Config fields

| Section | Key | Description |
|---------|-----|-------------|
| `global.grid` | `nx`, `ny`, `lx`, `ly` | Grid size and physical dimensions |
| `global.solver` | `dt`, `total_time`, `method` | Time-step, run length, solver backend |
| `global.physics` | standard constants | `hbar`, `m_eff`, `gamma_R`, `gamma_C`, `R`, … |
| `laser` | `sigma_space`, `sigma_time`, `pulse_separation`, `cutoff_sigma`, `n_pulses` | Single Gaussian pump laser (shared across all P values) |
| `sweep` | `P_min`, `P_max`, `P_step` | Power range and step |
| `sweep` | `scalar_check_every` | Record psi_sq_max every N steps (≥ 1) |
| `sweep` | `early_stop_on_divergence` | Terminate task immediately on NaN/Inf |
| `sweep` | `max_concurrent` | Max simultaneous GPU array tasks |
| `output` | `save_per_power_trace` | If true, write `power_<idx>_trace.npz` alongside each JSON |

## Run directory artifacts

```
<run_dir>/
├── config.yaml              frozen config copy
├── manifest.json            run metadata
├── power_index.json         [P_0, P_1, ..., P_N]
├── powers/
│   ├── power_000000.json    {P, status, psi_sq_max, t_psi_sq_max, …}
│   ├── power_000000_trace.npz  time / psi_sq_max arrays (if enabled)
│   └── …
├── threshold_curve.csv      summary table sorted by P
├── threshold_curve.json     same, JSON list
├── logs/
│   ├── sweep_<array>_<task>.out/err
│   └── finalize_<job>.out/err
└── results/
    └── psi_max_vs_power.png  plot: line for ok points, red ✗ for diverged
```

Each per-power JSON has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `P` | float | Pump power |
| `status` | `"ok"` / `"diverged"` | Simulation outcome |
| `psi_sq_max` | float | Global maximum of `|ψ|²` over the run |
| `t_psi_sq_max` | float | Time (ps) at which the max occurred |
| `diverged_at_step` | int / null | Step at which NaN/Inf was detected |
| `diverged_at_t` | float / null | Corresponding time (ps) |
| `wall_time_seconds` | float | GPU wall time for this task |
