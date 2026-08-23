# MNIST Digits Polariton SNN Dynamic

This is an experimental application example outside the WF-/WJ- requirements.
Polarism does not claim validated MNIST classification accuracy; its automated
end-to-end coverage checks execution and reproducibility only.

This workflow encodes downsampled MNIST digits into a 7x7 lattice of pulsed
Gaussian pumps and records dynamic condensate and reservoir traces for a
logistic-regression readout.

It is the ML workflow: scenario YAML files define encoding, lattice geometry,
pulse parameters, and readout settings, then the pipeline evaluates
classification. It does not run a threshold-search stage. Use
`threshold_finder` for pump-power response scans and `create_characteristic`
for two-dimensional threshold-crossing maps.

The Polarism physics configuration is shared through `polarism_base.yaml`.
Scenario YAML files provide the experiment-level settings: data selection, pump
geometry, pulse timing, encoder range, readout masks, classifier settings, and
output directory.

## Local Diagnostics

Campaign-specific diagnostics and profiling tools are local-only and excluded
from version control. They do not form part of this workflow's tracked
interface.

## Deterministic Initialization

Both single-sample and batched runners require `physics.init_seed` in the
resolved Polarism config. A missing or `null` seed fails fast instead of falling
back silently to `42`.

The shared base config currently sets:

```yaml
physics:
  init_seed: 42
```

Keep this explicit in production campaigns so single-image debugging and batched
runs start from physically equivalent initial conditions.

## Pitch/Sigma Sweep

The pitch/sigma campaign lives in:

```text
src/mnist_digits_polariton_snn_dynamic/scenarios/pitch_sigma_sweep/
```

Before submitting the campaign to GPU, validate the discrete Gaussian integral
against the analytic spatial Gaussian integral. The accepted threshold is
`1.0e-4` relative error. If a future scenario exceeds that threshold, do not
submit it directly. Either add an explicit `GridLatticeGeometry` sigma/dx guard
or refine the grid for the affected scenario and estimate the FFT cost increase.

Each campaign scenario is submitted as two dependent stages:

- `run`: GPU full MNIST run using `encoding.power_max` from its scenario YAML.
- `finalize`: CPU-only ablation readout that writes `ablation_report.json` and
  refreshes `campaign_ablation_summary.json`.

The submitters read the shared repository-root scheduler configuration. Create
it before the first submission with `cp slurm.env.example slurm.env`, then
replace the example cluster values and set any `SNN_MIX_*` resource overrides.

The baseline scenario is declared in the manifest as `baseline_scenario_id`.
For `pitch_sigma_sweep`, it is `pitch12_sigma2_baseline`. It is used only as
the reporting reference. The submitter runs scenarios sequentially: the run
for scenario `N+1` depends on successful finalize completion for scenario `N`.

Submit the campaign from the workflow directory with:

```bash
cd src/mnist_digits_polariton_snn_dynamic
bash cluster/submit_campaign.sh --dry-run scenarios/pitch_sigma_sweep/manifest.yaml
```

Pass `--with-calibrate` only when an explicit GPU calibration is wanted; the
run then uses `calibration.json` instead of its configured `power_max`.


## Ablation Output

The full run writes raw traces and audit inputs:

```text
features.npy
labels.npy
input_images.npy
encoded_powers.npy
traces_psi.npy
traces_nA.npy
traces_nI.npy
trace_times_ps.npy
```

Finalize recomputes the readout from raw traces. It compares:

- `pixel_baseline`: flattened downsampled input pixels.
- `full_features`: summary plus raw dynamic trace features.
- `field_stat_only`: all channel values for one field and one statistic.
- `field_all_stats`: all statistics for one field.
- `stat_all_fields`: one statistic across all recorded fields.

Each subset reports `accuracy_test` and `delta_vs_pixel_baseline`.
