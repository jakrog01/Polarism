# `pump_multi_comparison` Pipeline

`src/pump_multi_comparison/` is an example HPC workflow built on top of `polarism`. Its purpose is to automate a multi-scenario study on Slurm rather than provide new reusable package primitives.

## What it does

The pipeline takes a study configuration, creates a run directory, submits a Slurm DAG, executes GPU simulation stages, and then performs CPU-side visualization and summary steps.

At a high level, the workflow is:

1. validate the configuration and scheduler environment
2. run a threshold-search stage
3. launch one simulation job per scenario
4. generate per-scenario visualizations
5. aggregate results into a final summary

## Entry point

The supported entry point is:

```bash
bash src/pump_multi_comparison/submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run]
```

This wrapper owns the run-directory setup and Slurm submission logic. By default it writes runs under `src/pump_multi_comparison/runs/`; use `--runs-dir` to override that. The legacy scripts under `legacy/` are kept for reference only and are not part of the main path.

## Pipeline structure

| Part | Location | Responsibility |
| --- | --- | --- |
| Submission wrapper | `submit.sh` | validates inputs, prepares run directory, submits Slurm jobs |
| Cluster wrappers | `cluster/` | site-specific environment and resource setup |
| Config layer | `pipeline/config/` | YAML loading, validation, translation into `polarism.Config` |
| Manifest layer | `pipeline/manifest/` | run metadata and atomic JSON writes |
| Simulation core | `pipeline/simulation/core.py` | drives `polarism` and manages batched output |
| GPU stages | `pipeline/stages/gpu/` | threshold search and scenario execution |
| CPU stages | `pipeline/stages/cpu/` | plotting, visualization, and final aggregation |
| Legacy scripts | `legacy/` | deprecated paths kept outside the supported workflow |

## Slurm execution model

The pipeline is organized as a simple dependency graph:

```text
submit.sh
  -> threshold_search
  -> scenario array
  -> visualize array
  -> finalize
```

More precisely:

- `threshold_search` runs first and prepares the parameter region for the scenarios
- `run_scenario.py` executes one scenario per array task on GPU resources
- `visualize.py` creates per-scenario plots after successful scenario completion
- `finalize.py` builds cross-scenario summaries once all upstream work succeeds

## Typical outputs

Each run produces a timestamped directory containing:

- a frozen config snapshot
- scenario index and manifest JSON files
- one HDF5 file per scenario
- per-scenario metadata
- plots and aggregated summaries
- Slurm logs

With the default submission path, these live under `src/pump_multi_comparison/runs/<timestamp>_<config-hash>/`.

This makes the example useful as a reference for reproducible study orchestration, not just raw simulation execution.

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
