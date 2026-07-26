# MNIST Digits Polariton SNN Dynamic

This workflow encodes downsampled MNIST digits into a 7x7 lattice of pulsed
Gaussian pumps and records dynamic condensate and reservoir traces for a
logistic-regression readout.

The Polarism physics configuration is shared through `polarism_base.yaml`.
Scenario YAML files provide the experiment-level settings: data selection, pump
geometry, pulse timing, encoder range, readout masks, classifier settings, and
output directory.

## Local Diagnostics

Use the project virtual environment for every diagnostic:

```bash
.venv/bin/python src/mnist_digits_polariton_snn_dynamic/scripts/verify_geometry.py
.venv/bin/python src/mnist_digits_polariton_snn_dynamic/scripts/check_pitch_sigma_discretization.py
.venv/bin/python src/mnist_digits_polariton_snn_dynamic/scripts/profile_pump_allocation.py
```

`verify_geometry.py` checks that the current 7x7 lattice and linear encoder
match the old explicit implementation.

`check_pitch_sigma_discretization.py` is the CPU-only gate for
`scenarios/pitch_sigma_sweep/`. It evaluates each scenario Gaussian on the
configured real-space grid and compares the discrete integral against
`2*pi*sigma_space_um^2`. The current sweep has worst observed relative error
`3.18e-16` across all 7x7 lattice centers, including `pitch04_sigma0_67`, so no
sigma/dx validation threshold or per-scenario grid refinement is currently
required by this diagnostic.

`profile_pump_allocation.py` is an optional CuPy micro-profile for the nonzero
pulse branch in the time-step loop. The current measurement shows the
preallocated-buffer variant is not worth implementing for this workflow.

The current review record is kept in:

```text
src/mnist_digits_polariton_snn_dynamic/scripts/review_findings_report.md
```

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

Before submitting the campaign to GPU, run:

```bash
.venv/bin/python src/mnist_digits_polariton_snn_dynamic/scripts/check_pitch_sigma_discretization.py
```

The accepted diagnostic threshold for this campaign is `1.0e-4` relative error
against the analytic spatial Gaussian integral. If a future scenario exceeds
that threshold, do not submit it directly. Either add an explicit
`GridLatticeGeometry` sigma/dx guard or refine the grid for the affected
scenario and estimate the FFT cost increase.

Each campaign scenario is submitted as three dependent stages:

- `calibrate`: GPU full-lattice threshold sweep plus single-spot guard that
  writes `calibration.json`.
- `run`: GPU full MNIST run using `final_power_max` from that scenario's own
  calibration.
- `finalize`: CPU-only ablation readout that writes `ablation_report.json` and
  refreshes `campaign_ablation_summary.json`.

The baseline scenario is declared in the manifest as `baseline_scenario_id`.
For `pitch_sigma_sweep`, it is `pitch12_sigma2_baseline`. It is used only as
the reporting reference; it no longer defines calibrated powers for other
variants.

```text
run encoding.power_max = scenario calibration.final_power_max
```

The submitter runs scenarios sequentially. Calibration for scenario `N+1`
depends on successful finalize completion for scenario `N`.

Submit the campaign from the workflow directory with:

```bash
cd src/mnist_digits_polariton_snn_dynamic
bash cluster/submit_campaign.sh --dry-run scenarios/pitch_sigma_sweep/manifest.yaml
```

Do not submit without `--dry-run` until the condensation proxy used by
`simulation/calibration.py` is accepted.

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
