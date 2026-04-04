# pump_multi_comparison — Pipeline Architecture

## Entry point

```bash
bash submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run]
```

This is the **only supported entry point**.  It validates config, creates a
run directory, and submits the full Slurm DAG.  Legacy paths are in
`legacy/`.

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
│   │   ├── loader.py             ← YAML loading, power expression parsing
│   │   ├── validator.py          ← Pre-submit config + slurm.env validation
│   │   └── builder.py            ← Constructs polarism Config objects
│   ├── manifest/
│   │   └── io.py                 ← Atomic JSON writes, run manifest, scenario index
│   ├── simulation/
│   │   └── core.py               ← Physics kernel: AsyncBatchWriter, run_simulation_from_config
│   └── stages/
│       ├── gpu/
│       │   ├── threshold_search.py  ← GPU: find condensation threshold
│       │   └── run_scenario.py      ← GPU: simulate one scenario (array task)
│       └── cpu/
│           ├── viz_engine.py        ← Visualization engine (pure functions)
│           ├── visualize.py         ← CPU: per-scenario plots (array task)
│           └── finalize.py          ← CPU: cross-scenario summary
│
└── legacy/                       ← DEPRECATED — not used by the pipeline
    ├── DEPRECATED.md
    ├── orchestrate.sh
    ├── run_scenario.py
    └── optimize.py
```

---

## Slurm DAG

```
submit.sh  (login node — exits immediately after queuing)
    │
    ▼
[1] threshold_search   GPU  single job
    │  afterok
    ▼
[2] scenario array     GPU  --array=0-(N-1)%K   ← K = MAX_CONCURRENT_SCENARIOS
    │  afterok (ALL tasks must succeed)
    ├───────────────────────────────────┐
    ▼                                   ▼
[3] visualize array    non-GPU          [4] finalize   non-GPU
    per-scenario                             single job
```

- If any scenario GPU job fails, Slurm cancels jobs 3 and 4 via `afterok`
  dependency.  The run manifest shows which scenario failed.
- `--array=0-(N-1)%K` limits simultaneous scenario jobs to K.  Scenario
  temp directories are job-unique:
  `$SCRATCH/polariton/<run>/<SLURM_JOB_ID>_<TASK_ID>/<scenario>/`

---

## Run directories

Every `submit.sh` invocation creates:

```
runs/<YYYYMMDD_HHMMSS>_<config-hash>/
├── config.yaml              ← snapshot (jobs never read the live source)
├── scenario_index.json      ← ["single_pump", "double_pump", "cross_pattern"]
├── manifest.json            ← run state
├── threshold_result.json    ← written by job 1
├── <scenario>_meta.json     ← written by each scenario GPU job
├── <scenario>.h5            ← HDF5 output from GPU simulation
├── results_summary.json     ← written by finalize
├── results/                 ← plots and animations
└── logs/                    ← Slurm stdout/stderr per job
```

All writes to JSON metadata are atomic (temp → fsync → rename).

---

## Configuration reference

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
| `MAX_CONCURRENT_SCENARIOS` | optional | Max simultaneous scenario GPU jobs (default **1**) |
| `CPU_PARTITION` | optional | Partition for non-GPU stages — see below |
| `CPU_MEM` | optional | Memory for non-GPU stages (defaults to `SLURM_MEM`) |
| `CPU_CPUS` | optional | CPUs for non-GPU stages (defaults to `SLURM_CPUS`) |
| `CPU_TIME` | optional | Walltime for non-GPU stages (defaults to `SLURM_TIME`) |

### Non-GPU stage placement on Rysy

Rysy is a GPU cluster.  If `CPU_PARTITION` is not set (or equals
`SLURM_PARTITION`), non-GPU stages run on the GPU partition **without** a
GPU allocation.  `submit.sh` emits a WARNING in this case.

See `cluster/README.md` for the full resource-model discussion.

---

## Separation of concerns

| Layer | Location | Knows about |
|-------|----------|-------------|
| Config / schema | `pipeline/config/` | YAML format, dataclass fields |
| Manifest / atomic I/O | `pipeline/manifest/` | JSON, filesystem atomicity |
| Physics kernel | `pipeline/simulation/core.py` | polarism API, CuPy/NumPy |
| GPU stages | `pipeline/stages/gpu/` | kernel + config + manifest |
| CPU stages | `pipeline/stages/cpu/` | HDF5, matplotlib, manifest |
| Slurm submission | `submit.sh` | Slurm CLI, run-dir layout |
| Cluster wrappers | `cluster/` | Module names, PYTHONPATH |
