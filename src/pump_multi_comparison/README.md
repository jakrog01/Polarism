# pump_multi_comparison — Pipeline Architecture

## Entry point

```bash
bash submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run] [--wait]
```

This is the only supported entry point. It validates config, creates a run
directory, and submits the full Rysy workflow as Slurm jobs connected with
`afterok` dependencies. Legacy paths are in `legacy/`.

---

## Directory layout

```
pump_multi_comparison/
├── submit.sh                     ← ENTRY POINT
├── config.yaml                   ← Experiment configuration
│
├── cluster/                      ← Rysy Slurm job wrappers
│   ├── README.md                 ← Resource model for Rysy
│   ├── job_gpu.sh                ← GPU stage wrapper (loads CUDA)
│   └── job_stage.sh              ← Non-GPU stage wrapper (no CUDA)
│
├── pipeline/                     ← Python package
│   ├── config/
│   │   ├── loader.py             ← YAML loading, power/delay expression parsing
│   │   ├── validator.py          ← Pre-submit config + slurm.env validation
│   │   └── builder.py            ← Constructs configs + resolves per-laser delays
│   ├── manifest/
│   │   └── io.py                 ← Atomic JSON writes, run manifest, scenario index
│   ├── simulation/
│   │   └── core.py               ← Physics kernel: auto-sized appendable HDF5 writer + run_simulation_from_config
│   └── stages/
│       ├── gpu/
│       │   ├── threshold_search.py  ← GPU: find condensation threshold
│       │   └── run_scenario.py      ← GPU: simulate one scenario + inline render
│       └── cpu/
│           ├── viz_engine.py        ← Visualization engine (pure functions)
│           ├── visualize.py         ← Optional post-hoc plots if raw HDF5 was archived
│           └── finalize.py          ← CPU: cross-scenario summary from lightweight artifacts
│
└── legacy/                       ← DEPRECATED — not used by the pipeline
    ├── DEPRECATED.md
    ├── orchestrate.sh
    ├── run_scenario.py
    └── optimize.py
```

---

## Execution model

```
submit.sh  (Rysy login node)
    │
    ├── sbatch threshold_search
    ├── sbatch scenario array  --dependency=afterok:<threshold_job>  --array=0-(N-1)%K
    │       each task: simulate on NVMe scratch -> render inline -> copy back artifacts
    └── sbatch finalize        --dependency=afterok:<scenario_array_job>
```

- `submit.sh` is asynchronous by default: once all jobs are accepted by Slurm,
  the terminal can be closed.
- Pass `--wait` to restore synchronous polling and abort locally on the first
  failed Rysy job.
- `--array=0-(N-1)%K` limits simultaneous scenario jobs to K.  Scenario
  temp directories are job-unique:
  `$SCRATCH/polariton/<run>/<SLURM_JOB_ID>_<TASK_ID>/<scenario>/`

---

## Run directories

Every `submit.sh` invocation creates a run directory under `TETYDA_RUNS_BASE`
unless `--runs-dir` overrides it:

```
<runs-base>/<YYYYMMDD_HHMMSS>_<config-hash>/
├── config.yaml              ← snapshot (jobs never read the live source)
├── scenario_index.json      ← ["seq_5spot_p1.0", "seq_5spot_p0.9", ...]
├── manifest.json            ← run state
├── threshold_result.json    ← written by job 1
├── <scenario>_meta.json     ← written by each scenario GPU job
├── <scenario>_scalars.npz   ← scalar sidecar for finalize
├── <scenario>.h5            ← optional raw archive when archive_raw_hdf5=true
├── results_summary.json     ← written by finalize
├── results/                 ← plots and animations
└── logs/                    ← Slurm stdout/stderr per job
```

All writes to JSON metadata are atomic (temp → fsync → rename).

---

## Scenario schema

`config.yaml` now supports reusable same-file YAML anchors for scenario baselines.
The current example defines:

- shared `timing_vars` expressions evaluated from threshold-search output
- a reusable `lasers` template with explicit per-laser `delay`
- one scenario per power level, using `power_modifiers` to scale selected laser IDs
- per-laser finite train length via `n_pulses`

Scenario timing is deterministic:

- each laser may define `delay` as either a number or an arithmetic expression
- allowed delay-expression names start with `sigma_time`, `pulse_separation`, and `cutoff_sigma`
- `timing_vars` can introduce readable derived quantities such as `pulse_duration`
- in the current sequential 5-spot example, `pulse_duration` is the full finite
  pulse support width `2 * cutoff_sigma * sigma_time`, not the half-width
- `pulse_separation` may be expressed directly on a laser or derived from threshold
  output through `timing_vars`
- finite pulse trains use `n_pulses`; after the train is exhausted, that laser stops
  contributing pump
- the legacy relational `timing:` block is no longer supported

Scenario power remains threshold-relative:

- per-laser `power` accepts numeric literals and `P`-relative forms such as `1.0P` and `0.6P`
- scenario-level `power_modifiers` can target explicit laser IDs
- tags remain optional metadata and can also be matched, but IDs are the primary targeting path
- the finalize-stage `summary.png` contains comparison traces only; field snapshots
  are written as per-scenario plots instead of being embedded in the summary sheet

## Threshold Search

Threshold search is no longer just a single centered-spot proxy by default.

- the optimization reference is always the first scenario in `config.yaml`
- the search reuses that scenario's laser geometry, delays, power modifiers, finite
  train counts, and potential
- the retained operating point is chosen lexicographically:
  1. minimum power
  2. minimum `sigma_time`
  3. minimum condensation time `t_cond`
  4. minimum `pulse_separation` when an explicit separation list is used

This keeps the threshold search aligned with the actual protocol that the run will
simulate instead of an unrelated one-spot approximation.

Threshold-search timing can be configured in two ways:

- explicit search grid via `pulse_separation_values`
- derived cadence via `pulse_separation_formula`

When `pulse_separation_formula` is used, each `sigma_time` maps to one derived
separation and there is no independent separation sweep. In the current sequential
example:

- `pulse_duration = 2 * cutoff_sigma * sigma_time`
- `cycle_duration = 4 * pulse_duration`
- `pulse_separation_formula = 8 * cutoff_sigma * sigma_time`

The scenario timing-budget validator checks the full finite train, not just the
first pulse support window. For each laser the latest required end time is:

`delay + (n_pulses - 1) * pulse_separation + 2 * cutoff_sigma * sigma_time`

For generalized scenario support, threshold search is deterministic even for legacy
scenarios that still rely on random phase offsets: each candidate evaluation starts
from the same fixed RNG seed, so results do not depend on search order.

---

## Configuration reference

### `global.threshold_search`

| Key | Required | Description |
|-----|----------|-------------|
| `power_values` | ✓ | Candidate threshold powers scanned in ascending order |
| `sigma_time_values` | ✓ | Candidate pulse durations scanned in ascending order |
| `pulse_separation_values` | conditional | Explicit candidate separations; use this or `pulse_separation_formula` |
| `pulse_separation_formula` | conditional | Arithmetic expression deriving separation from `sigma_time` / `cutoff_sigma` |
| `n_pulses` | optional | Finite train length for the search kernel when the single-laser path is used; scenario-based search uses per-laser values from scenario 0 |
| `condensation_fraction` | ✓ | Fraction of `global.solver.total_time` allocated to the search stage |
| `dt_multiplier` | optional | Coarsening factor for the search-stage timestep |
| `max_runtime_minutes` | ✓ | Wall-clock cap for the search stage |

### `slurm.env` (required)

| Variable | Required | Description |
|----------|----------|-------------|
| `SLURM_ACCOUNT` | ✓ | Slurm account/project |
| `SLURM_PARTITION` | ✓ | GPU partition |
| `SLURM_MEM` | ✓ | Memory for GPU jobs (e.g. `64G`) |
| `SLURM_GPUS` | ✓ | GPUs per job (e.g. `1`) |
| `SLURM_CPUS` | ✓ | CPUs per GPU job |
| `SLURM_TIME` | ✓ | Walltime for scenario jobs (e.g. `04:00:00`) |
| `SLURM_QOS` | optional | QOS string |
| `NVME_GB` | ✓ | NVMe scratch size requested for scenario jobs |
| `TETYDA_RUNS_BASE` | ✓ | Persistent run base on `/lu/tetyda` |
| `MAX_CONCURRENT_SCENARIOS` | ✓ | Max simultaneous scenario GPU jobs |
| `FINALIZE_MEM` | ✓ | Memory for the finalize stage |
| `FINALIZE_CPUS` | ✓ | CPUs for the finalize stage |
| `FINALIZE_TIME` | ✓ | Walltime for the finalize stage |

### Scratch directory requirements

Each scenario GPU job writes its transient HDF5 output (several GB before compression) to a
job-unique subdirectory under the first usable scratch base it finds, probed in
this order: `$SCRATCH`, `$SLURM_TMPDIR`, `$TMPDIR`.  The resulting path looks like:

```
<base>/polariton/<run>/<SLURM_JOB_ID>_<TASK_ID>/<scenario>/
```

Any candidate whose canonical path (`realpath`) is exactly `/tmp` is skipped.  If
all three variables are unset, absent, or resolve to `/tmp`, and the job is running
under Slurm, it fails at scratch-directory creation with:

```
No suitable scratch directory found for the GPU scenario job.
Set and export SCRATCH in slurm.env or provide SLURM_TMPDIR/TMPDIR
that points to real job scratch, not /tmp.
```

**Current Rysy flow (`submit.sh`):** Do not set `SCRATCH` in `slurm.env`.
`job_gpu.sh` exports `SCRATCH="/scratch/${SLURM_JOBID}"` at job runtime, using the
NVMe-oF device allocated via `--gres=nvme:${NVME_GB}`.  Setting `SCRATCH` in
`slurm.env` would override this with a path that does not exist inside the job.

**Failure mode before this guard:** when `/tmp` was accepted as scratch (it was,
because the guard only applied to `SLURM_TMPDIR`/`TMPDIR` but not `SCRATCH`), the
job would complete 100 % of the solver loop and then fail during `writer.close()` /
HDF5 final flush as `/tmp` filled, exiting with code 1 and no `→ file.h5` line in
stdout.  With the fix the job fails before the simulation starts.

---

## Separation of concerns

| Layer | Location | Knows about |
|-------|----------|-------------|
| Config / schema | `pipeline/config/` | YAML format, dataclass fields |
| Manifest / atomic I/O | `pipeline/manifest/` | JSON, filesystem atomicity |
| Physics kernel | `pipeline/simulation/core.py` | polarism API, CuPy/NumPy, appendable HDF5 writer selection |
| GPU stages | `pipeline/stages/gpu/` | kernel + config + manifest |
| CPU stages | `pipeline/stages/cpu/` | scalar sidecars, summary plots, optional post-hoc HDF5 visualisation |
| Slurm submission | `submit.sh` | local sbatch submission/dependencies on a Rysy login node, run-dir layout |
| Cluster wrappers | `cluster/` | Module names, PYTHONPATH |
