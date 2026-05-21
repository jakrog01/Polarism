# `pump_multi_comparison` Pipeline

`src/pump_multi_comparison/` is an example HPC workflow built on top of `polarism`. Its purpose is to automate a multi-scenario study on Slurm rather than provide new reusable package primitives.

## What it does

The pipeline takes a study configuration, creates a run directory, submits Rysy jobs locally from a Rysy login node, executes GPU simulation stages, renders per-scenario artifacts inline, and then performs a lightweight final summary step.

At a high level, the workflow is:

1. validate the configuration and scheduler environment
2. run a threshold-search stage
3. launch one simulation job per scenario
4. stream the per-scenario animation during simulation and render PNG artifacts afterward
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
- optional per-scenario `dynamics.mp4` or `dynamics.mkv` files
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

## Streaming animation path

Scenario movies are now rendered through the reusable
`polarism.results` animation stack rather than by a post-hoc HDF5 reader.  When
`output.render_animation: true`, `run_scenario.py` creates an
`AnimationVisitor` before the simulation starts and passes it into the
simulation core.  Frames are streamed at the same cadence as field snapshots.

This path is designed to avoid the previous failure mode where long jobs could
finish the numerical simulation and then hang inside the animation renderer.

Important operational details:

- ffmpeg is resolved from `FFMPEG_BIN` first, then `PATH`
- `RENDER_ENCODER` is respected when set
- the selected encoder runs a real preflight encode before any simulation work
- encoder stderr is written to a temporary file instead of a pipe, avoiding pipe-buffer deadlocks
- animation errors abort only the animation visitor; HDF5 and scalar sidecars are still finalized
- scenario metadata records `animation_status` and `animation_error`
- if animation was required and failed, the scenario exits with code `2` after artifact copy-back

The default panel set is:

| Panel | Source | Transform |
| --- | --- | --- |
| `psi` | condensate wavefunction | `abs2` |
| `nA` | active reservoir, stored under this legacy name for all reservoir models | none |
| `nI` | inactive reservoir | none |
| `Pump` | total pump field | power-normalized colour map |

For `quadratic-double`, the reusable `polarism` model calls the active reservoir
`nR`, but the pipeline writes that active field as `nA` in HDF5 and animations
for backward compatibility.

For delayed-growth or multi-pulse cases, fixed colour limits are recommended:

```yaml
output:
  render_animation: true
  animation_clim:
    psi: [0.0, 1.0]
    nA: [0.0, 10.0]
    nI: [0.0, 100.0]
```

Bad `animation_clim` entries produce a warning and are ignored.

## Scenario configuration

The current `src/pump_multi_comparison/config.yaml` example uses a deterministic
multi-laser scenario schema built around reusable YAML anchors:

- `timing_vars` defines shared arithmetic expressions such as `pulse_duration`
  and `cycle_duration`, evaluated from threshold-search results
- each laser defines an explicit `delay` as either a number or an arithmetic
  expression
- `power_modifiers` scales selected laser IDs using threshold-relative power
  expressions such as `0.9P`
- `laser_defaults.power_definition` controls whether pulsed Gaussian `power`
  values are local peak amplitudes or integrated per-pulse doses

This keeps the scenario description readable while remaining flexible for
regular or irregular spatial layouts. The older relational `timing:` syntax is
not part of the current schema.

For geometry-sensitive pulsed campaigns, use:

```yaml
global:
  laser_defaults:
    laser_type: pulse-gaussian
    power_definition: pulse_energy
```

In this mode, `parameter_sweep.power_values`, per-laser `power`, and
`power_modifiers` all scale the integrated dose of each pulse.  The local
centre amplitude is derived from the spot size and pulse duration.  This avoids
the common error where increasing `sigma_space` at fixed peak amplitude also
increases the total injected reservoir population.

For pulsed Gaussian lasers, `n_pulses` has two distinct meanings:

- `n_pulses > 0`: finite train length; the laser stops after that many pulses
- `n_pulses: 0`: unbounded pulse train; used intentionally in memory-response
  studies

This is important when interpreting repeated reservoir excitation. A run with
`n_pulses: 0` is not a single-pulse experiment.

## Current artifact-diagnostics workflow

Local validation campaigns are kept under `src/pump_multi_comparison/scenarios/`
and ignored by Git.  One such local campaign,
`scenarios/config_artifact_mitigation_validation.yaml`, diagnoses the diagonal
X/star-like spatial pattern that can appear in `|psi|^2`.

The config does not change the physical GPE during time evolution. It compares:

- legacy positive-uniform seed vs filtered zero-mean seed
- 128 um / 512^2 vs 128 um / 1024^2 at fixed physical domain
- 64 um / 512^2 as a cheaper high-resolution production domain
- five-point vs isotropic 9-point Laplacian in `rk4-cuda`

Run it from a Rysy login node:

```bash
cd ~/polaritonSNN/PolaritonSNN/src/pump_multi_comparison
bash submit.sh --config scenarios/config_artifact_mitigation_validation.yaml --dry-run
bash submit.sh --config scenarios/config_artifact_mitigation_validation.yaml
```

Primary artifacts:

| File | Purpose |
| --- | --- |
| `psi_sq.png` | real-space condensate-density snapshots, including peak frame |
| `psi_k.png` | log k-space power snapshots |
| `nA.png`, `nI.png` | active/inactive reservoir fields for `quadratic-double` |
| `<scenario>_scalars.npz` | scalar traces, including high-k metrics and pump-dose integrals |
| `<scenario>_meta.json` | grid, laser, solver, initial-condition, and effective pump-definition metadata |

Interpretation:

- if `filtered_seed` improves over `baseline`, the old biased/unfiltered seed was
  a major contributor
- if `same_domain_1024` improves over `filtered_seed`, sampling at fixed physical
  domain matters
- if `small_domain` improves, it may be a useful production domain, but remember
  that CAP/boundary distance also changed
- if `*_9pt` improves over its five-point counterpart, stencil anisotropy was
  significant
- if high-k power still accumulates near Nyquist after filtered seed, higher
  resolution, and 9-point stencil, the missing ingredient is likely a physical
  relaxation/high-k damping mechanism, not a higher-order RK time integrator

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
