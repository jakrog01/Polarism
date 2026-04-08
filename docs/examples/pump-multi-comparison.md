# `pump_multi_comparison` Pipeline

`src/pump_multi_comparison/` is an example HPC workflow built on top of `polarism`. Its purpose is to automate a multi-scenario study on Slurm rather than provide new reusable package primitives.

## What it does

The pipeline takes a study configuration, creates a run directory, submits Rysy jobs locally from a Rysy login node, executes GPU simulation stages, renders per-scenario artifacts inline, and then performs a lightweight final summary step.

At a high level, the workflow is:

1. validate the configuration and scheduler environment
2. run a threshold-search stage
3. launch one simulation job per scenario
4. render per-scenario plots and animations inline in each scenario job
5. aggregate results into a final summary

## Entry point

The supported entry point is:

```bash
bash src/pump_multi_comparison/submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run]
```

This wrapper owns the run-directory setup and stage polling logic on a Rysy login node. By default it writes runs under `TETYDA_RUNS_BASE`; use `--runs-dir` to override that. The legacy scripts under `legacy/` are kept for reference only and are not part of the main path.

## Pipeline structure

| Part | Location | Responsibility |
| --- | --- | --- |
| Submission wrapper | `submit.sh` | validates inputs, prepares run directory, submits and polls Rysy jobs |
| Cluster wrappers | `cluster/` | site-specific environment and resource setup |
| Config layer | `pipeline/config/` | YAML loading, validation, translation into `polarism.Config` |
| Manifest layer | `pipeline/manifest/` | run metadata and atomic JSON writes |
| Simulation core | `pipeline/simulation/core.py` | drives `polarism` and delegates HDF5 output to reusable storage backends |
| GPU stages | `pipeline/stages/gpu/` | threshold search and scenario execution |
| CPU stages | `pipeline/stages/cpu/` | final aggregation and optional post-hoc visualization from archived HDF5 |
| Legacy scripts | `legacy/` | deprecated paths kept outside the supported workflow |

## Slurm execution model

The pipeline is organized as a simple synchronous flow:

```text
submit.sh
  -> sbatch threshold_search
  -> wait
  -> sbatch scenario array
  -> wait
  -> sbatch finalize
```

More precisely:

- `threshold_search` runs first and prepares the parameter region for the scenarios
- `run_scenario.py` executes one scenario per array task on GPU resources, renders outputs inline, and copies back lightweight artifacts
- `visualize.py` is a post-hoc helper used only when raw HDF5 archival is enabled
- `finalize.py` builds cross-scenario summaries from metadata and scalar sidecars once all upstream work succeeds

## Typical outputs

Each run produces a timestamped directory containing:

- a frozen config snapshot
- scenario index and manifest JSON files
- one scalar sidecar per scenario
- per-scenario metadata
- plots and aggregated summaries
- optional raw HDF5 files only if archival is enabled
- Slurm logs

With the default submission path, these live under `TETYDA_RUNS_BASE/<timestamp>_<config-hash>/`.

This makes the example useful as a reference for reproducible study orchestration, not just raw simulation execution.

## HDF5 write path

Per-scenario HDF5 output is now created through `polarism.results.storage.create_hdf5_writer(...)` instead of a pipeline-local writer implementation.

- CPU runs use a buffered append-only writer.
- GPU runs use a double-buffered async writer that stages device data through host buffers.
- Batch depth is computed inside the simulation core from the declared output fields and the current grid shape, so `run_scenario.py` no longer computes or passes a batch size itself.

This keeps storage mechanics in the library and leaves the pipeline responsible for scheduling, manifests, and scenario-level orchestration.

## Scenario configuration

The current `src/pump_multi_comparison/config.yaml` example uses a deterministic
multi-laser scenario schema built around reusable YAML anchors:

- `timing_vars` defines shared arithmetic expressions such as `pulse_duration`
  and `cycle_duration`, evaluated from threshold-search results
- each laser defines an explicit `delay` as either a number or an arithmetic
  expression
- `power_modifiers` scales selected laser IDs using threshold-relative power
  expressions such as `0.9P`

This keeps the scenario description readable while remaining flexible for
regular or irregular spatial layouts. The older relational `timing:` syntax is
not part of the current schema.

## Why it lives in `src/`

The pipeline mixes domain-specific orchestration concerns with package usage:

- Slurm submission
- cluster environment setup
- run-directory conventions
- study-specific threshold search
- batch visualization and aggregation

Those concerns are valuable examples, but they should stay separate from the reusable physics and solver API in `polarism/`.

## When to use it

Use this example when you want a template for:

- multi-scenario parameter studies
- Slurm-based GPU scheduling
- separating simulation, visualization, and final aggregation stages
- organizing reproducible campaign outputs

If you only need to run a single simulation locally, start with the package-level quickstart instead of this pipeline.
