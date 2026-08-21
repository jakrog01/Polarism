# `polariton_hpc_pipeline` Pipeline

`src/polariton_hpc_pipeline/` is an example HPC workflow built on top of `polarism`. Its purpose is to automate a multi-scenario study on Slurm rather than provide new reusable package primitives.

## What it does

The pipeline takes a study configuration, creates a run directory, submits Rysy jobs locally from a Rysy login node, executes GPU simulation stages, renders per-scenario artifacts inline, and then performs a lightweight final summary step.

There are two supported run modes:

- `threshold_search`: run a calibration search first, then evaluate scenarios using the threshold result
- `parameter_sweep`: expand scenarios directly over configured axes and skip threshold search by writing a synthetic threshold stub

At a high level, the workflow is:

1. validate the configuration and scheduler environment
2. freeze an expanded run-local config and scenario index
3. run threshold search when the config is not in `parameter_sweep` mode
4. launch one simulation job per scenario
5. render PNG, animation, ROI, and optional fringe artifacts from node-local scratch
6. aggregate results into final summaries and sweep diagnostics

## Entry point

The supported entry point is:

```bash
bash src/polariton_hpc_pipeline/submit.sh [--config config.yaml] [--runs-dir /path/to/runs] [--dry-run]
```

This wrapper owns the run-directory setup and stage polling logic on a Rysy login node. By default it writes runs under `TETYDA_RUNS_BASE`; use `--runs-dir` to override that.

## Pipeline structure

| Part | Location | Responsibility |
| --- | --- | --- |
| Submission wrapper | `submit.sh` | validates inputs, prepares run directory, submits and polls Rysy jobs |
| Cluster wrappers | `cluster/` | site-specific environment and resource setup |
| Config layer | `pipeline/config/` | YAML loading, validation, translation into `polarism.Config` |
| Parameter sweep layer | `pipeline/config/sweep.py` | direct expansion over power, pulse separation, pulse width, spot size, and square-side axes |
| Output-policy layer | `pipeline/config/output_policy.py` | resolves raw-output and artifact-retention policy |
| Sweep utilities | `pipeline/config/sweep_utils.py` | shared expansion and laser-type resolution helpers |
| Experiment layer | `pipeline/experiments/` | registry plus `generic`, `probe5_trap_gate`, and `square4_fringe` experiment definitions |
| Manifest layer | `pipeline/manifest/` | run metadata and atomic JSON writes |
| Simulation core | `pipeline/simulation/core.py` | drives `polarism` and delegates HDF5 output to reusable storage backends |
| ROI utilities | `pipeline/simulation/roi.py` | region-of-interest definitions and reductions |
| Analysis layer | `pipeline/analysis/` | NumPy-only post-run metrics such as square-4 fringe analysis |
| Fringe analysis | `pipeline/analysis/fringe.py` | square-4 fringe metrics |
| Render layer | `pipeline/render/nvenc_stream.py` | streaming FFmpeg/NVENC encoding |
| GPU stages | `pipeline/stages/gpu/` | threshold search and scenario execution |
| CPU stages | `pipeline/stages/cpu/` | final aggregation and optional post-hoc visualization from archived HDF5 |

## Slurm execution model

The threshold-search mode is organized as:

```text
submit.sh
  -> sbatch threshold_search
  -> sbatch scenario array + last-scenario singleton
  -> sbatch finalize
```

The direct-sweep mode is shorter:

```text
submit.sh
  -> prepare_run writes threshold_result.json stub
  -> sbatch scenario array + last-scenario singleton
  -> sbatch finalize
```

More precisely:

- `threshold_search` runs first and prepares the parameter region for the scenarios
- `parameter_sweep` skips `threshold_search`; expanded scenario powers are already absolute in the frozen `config.yaml`
- `run_scenario.py` executes one scenario per array task or singleton on GPU resources, renders outputs inline, and copies back lightweight artifacts
- `visualize.py` is a post-hoc helper used only when raw HDF5 archival is enabled
- `finalize.py` builds cross-scenario summaries from metadata and scalar sidecars once all upstream work succeeds

The last scenario is submitted as a singleton so the project does not rely on
the Rysy Slurm array parent task for real GPU work.

## Typical outputs

Each run produces a timestamped directory containing:

- a frozen config snapshot
- scenario index and manifest JSON files
- one scalar sidecar per scenario
- optional ROI trace PNGs when ROI metrics are configured
- optional fringe JSON sidecars for square/interference studies
- per-scenario metadata
- plots and aggregated summaries
- parameter-sweep heatmaps and diagnostic maps when sweep metadata is present
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

## Scenario animation path

Scenario movies are now rendered through the reusable
`pipeline.render.nvenc_stream` HDF5 renderer while the HDF5 file is still on
node-local scratch.  When `output.render_animation: true`, `run_scenario.py`
first completes the numerical simulation and closes the HDF5 writer, then the
renderer scans all recorded frames to compute global colour limits before
streaming RGB frames into ffmpeg.

This two-pass post-render path avoids early-frame autoscaling: delayed
condensation in multi-pulse runs no longer saturates the movie just because the
first frames were nearly empty.

Important operational details:

- ffmpeg is resolved from `FFMPEG_BIN` first, then `PATH`
- `RENDER_ENCODER` is respected when set
- encoder stderr is written to a temporary file instead of a pipe, avoiding pipe-buffer deadlocks
- colour limits are scanned over the full HDF5 trajectory; low-density values use physical zero as black
- animation errors happen after HDF5, sidecar, and PNG generation; these artifacts are copied back before the scenario exits nonzero
- scenario metadata records `animation_status` and `animation_error`
- if animation was required and failed, the scenario exits with code `2` after artifact copy-back

The default panel set is:

| Panel | Source | Transform |
| --- | --- | --- |
| `psi` | condensate wavefunction | `abs2` |
| `nA` | active reservoir, stored under this legacy name for all reservoir models | none |
| `nI` | inactive reservoir | none |

For `quadratic-double`, the reusable `polarism` model calls the active reservoir
`nR`, but the pipeline writes that active field as `nA` in HDF5 and animations
for backward compatibility.

`Pump` is intentionally not part of the default movie panel set.  Narrow pulses
can be much shorter than the field-record cadence, so a movie frame may sample
an arbitrary phase of the pulse and visually under-report the true pump peak.
Use `Pump.png`, `P_max`, `P_area_integral`, and
`P_cumulative_area_time_integral` for pump diagnostics.

Automatic global colour limits are the default.  Fixed limits can still be set
when comparing multiple movies on exactly the same visual scale:

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

The current `src/polariton_hpc_pipeline/config.yaml` example is a GaAs/AlGaAs
single-spot pulsed campaign for the `quadratic-double` reservoir model.  It is
not a dimensionless toy preset: the interaction constants, lifetime scales,
pump normalization, spot size, and solver choice are intended to be read
together.

The tracked default uses `global.parameter_sweep.enabled: true`, so submission
skips the threshold-search GPU stage.  `prepare_run` expands the configured base
scenarios into a frozen run-local `config.yaml`, writes `scenario_index.json`,
and writes a synthetic `threshold_result.json` with `mode: parameter_sweep`.
That stub exists only to keep the downstream scenario builder on the same
runtime path as threshold-calibrated runs.

### Default GaAs spot sweep

The default campaign uses a 4-quantum-well GaAs microcavity estimate with
exciton fraction \(|X|^2 = 0.40\).  The condensate and reservoir interaction
constants are derived from the 2D exciton estimate
\(g_{xx} = 6 E_b a_B^2\), using \(E_b = 10\,\mathrm{meV}\) and
\(a_B = 0.01\,\mu\mathrm{m}\), then scaled by the Hopfield coefficient and
`N_QW = 4`.

| Parameter | Value in `config.yaml` | Meaning |
| --- | ---: | --- |
| `m_eff` | `0.32` | lower-polariton effective mass in meV ps^2 / um^2 |
| `gamma_C` | `0.1` | condensate decay rate, about 10 ps lifetime |
| `gamma_R` | `0.15` | active-reservoir decay rate |
| `gamma_I` | `0.001` | slow inactive-reservoir decay |
| `g_C` | `0.00024` | condensate self-interaction in meV um^2 |
| `g_R` | `0.0006` | active-reservoir blueshift in meV um^2 |
| `g_I` | `0.0` | inactive-reservoir blueshift disabled in this preset |
| `R` | `0.023` | stimulated scattering rate in ps^-1 um^2 |
| `kappa` | `0.05` | quadratic inactive-to-active transfer rate |

These values are a GaAs working preset for the current pulsed protocol, not a
claim that every coefficient has a unique material value.  In particular,
`gamma_I` and `kappa` are phenomenological reservoir-memory parameters and
should be swept if the burst dynamics are the conclusion of the study.

The pump is normalized with:

```yaml
global:
  laser_defaults:
    laser_type: pulse-gaussian
    sigma_space: 2.0
    sigma_time: 1.7
    power_definition: pulse_energy
```

The scenario powers (`3500`, `5000`, `8000`, `13000`, `20000`) are absolute
integrated per-pulse doses.  Because those powers live on the scenarios,
`parameter_sweep.power_values: [1.0]` is only a one-point sweep placeholder.
During expansion, absolute per-laser powers are used in scenario names and
metadata, so the default run expands to names such as `gentle_E3500_sep12` and
`burst_E20000_sep12`.

For threshold-relative or percentage-style powers (`P`, `0.6P`, `1.2P`, and
the same forms in `power_modifiers`), the sweep label remains the swept
reference value.  For example, a laser with `power: "0.6P"` at
`parameter_sweep.power_values: [100.0]` is named with `E100`, while the resolved
laser dose is `60.0`.

Source context for the GaAs preset:

- Comaron et al., "Coherence of a non-equilibrium polariton condensate across
  the interaction-mediated phase transition", Communications Physics 8, 94
  (2025), https://doi.org/10.1038/s42005-025-01977-7.  This provides recent
  GaAs/AlGaAs context for excitonic-fraction-dependent interactions and
  driven-dissipative simulations.
- Gnusov et al., "Quantum vortex formation in the rotating bucket experiment
  with polariton condensates", Science Advances 9, eadd1299 (2023),
  https://doi.org/10.1126/sciadv.add1299.  This is a GaAs polariton-flow
  experiment modeled with a generalized Gross-Pitaevskii equation.
- Hu, Deng, and Liu, "Two-dimensional exciton-polariton interactions beyond
  the Born approximation", Physical Review A 106, 063303 (2022),
  https://doi.org/10.1103/PhysRevA.106.063303.  This is the reference for the
  limits of the Born-style interaction estimate used to set the scale here.
- Sun et al., "Direct measurement of polariton-polariton interaction
  strength", Nature Physics 13, 870-875 (2017),
  https://doi.org/10.1038/nphys4148.  This is useful context for the magnitude
  and ambiguity of GaAs interaction-strength measurements.
- Pieczarka et al., "Effect of optically induced potential on the energy of
  trapped exciton polaritons below the condensation threshold", Physical Review
  B 100, 085301 (2019), https://doi.org/10.1103/PhysRevB.100.085301.  This is
  a cautionary reference for interpreting blueshifts when reservoir-induced
  potentials are present.

### YAML structure

The scenario schema supports deterministic single- or multi-laser studies built
around reusable YAML anchors:

- `timing_vars` defines shared arithmetic expressions such as `pulse_duration`
  and `cycle_duration`, evaluated from threshold-search results
- each laser defines an explicit `delay` as either a number or an arithmetic
  expression
- `power_modifiers` scales selected laser IDs using threshold-relative power
  expressions such as `0.9P`
- `laser_defaults.power_definition` controls whether pulsed Gaussian `power`
  values are local peak amplitudes or integrated per-pulse doses
- `parameter_sweep` can expand `power_values`, `pulse_separation_values`,
  `sigma_time_values`, `sigma_space_values`, and selected `base_scenarios`

This keeps the scenario description readable while remaining flexible for
regular or irregular spatial layouts. The older relational `timing:` syntax is
not part of the current schema.

For `geometry: square4`, the sweep layer treats `square_side_values` as the
spatial axis.  The first four lasers are moved to
`(-a/2,-a/2)`, `(a/2,-a/2)`, `(-a/2,a/2)`, and `(a/2,a/2)` for each side
length `a`; scenario names use an `_a..._E...` or `_a..._P...` suffix instead
of `_sep...`.  The first pulse-separation value is still applied as fixed pulse
timing and recorded in scenario metadata.

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

### ROI and fringe diagnostics

The output block can request circular ROI metrics:

```yaml
output:
  roi_metrics:
    enabled: true
    rois:
      - id: core_1sigma
        shape: circle
        x0: 0.0
        y0: 0.0
        radius: "sigma_space"
```

ROI radii may be numeric or expressions resolved against the first laser and
scenario namespace.  Scalar sidecars then include ROI traces such as mean
`|psi|^2`, integrated `|psi|^2`, reservoir means, and an emission proxy.  The
PNG renderer writes `roi_traces.png` next to each scenario's field snapshots.

For interference-style square studies, enable:

```yaml
output:
  fringe_analysis:
    enabled: true
    fringe_window_radius: 16.0
```

`run_scenario.py` computes the fringe sidecar after HDF5 output and rendering,
while the file is still on node-local scratch.  Pump cores are excluded using
`cutoff_sigma * sigma_space`, frames before the pulse train has ended are
skipped, and the finalizer writes:

- `<scenario>_fringe.json` per scenario
- `results/spatiotemporal_square4_fringe_summary.csv`
- `results/spatiotemporal_square4_fringe_summary.json`
- `results/selected_extremes.json`

The fringe metrics include robust contrast, coefficient of variation, dominant
FFT fringe spacing, horizontal/vertical line-scan contrast, central ROI peaks,
and a `crossed_threshold` flag using `psi_sq_max >= 5e-2` inside the analysis
window.

### Square-4 spatiotemporal preflight

`src/polariton_hpc_pipeline/scenarios/config_spatiotemporal_square4.yaml` is the
current square-4 preflight for the spiking-paper campaign.  It uses four
synchronous pulsed Gaussian spots on the corners of a square, scans square side
length from `8.0` to `32.0 um`, and evaluates pulse energies
`1600`, `2000`, and `2400` in `pulse_energy` mode.

Run it from a Rysy login node:

```bash
# Set POLARISM_ROOT to the Polarism checkout root on the cluster.
cd $POLARISM_ROOT/src/polariton_hpc_pipeline
bash submit.sh --config scenarios/config_spatiotemporal_square4.yaml --dry-run
bash submit.sh --config scenarios/config_spatiotemporal_square4.yaml
```

This config is a parameter-sweep preflight, not a threshold-calibrated protocol:
threshold search is skipped, the square side is the sweep axis, and the useful
outputs are the scalar sweep heatmaps, ROI traces, and fringe summaries.

## Current artifact-diagnostics workflow

Local validation campaigns are kept under `src/polariton_hpc_pipeline/scenarios/`
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
# Set POLARISM_ROOT to the Polarism checkout root on the cluster.
cd $POLARISM_ROOT/src/polariton_hpc_pipeline
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
