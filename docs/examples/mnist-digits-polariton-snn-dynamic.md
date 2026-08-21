# mnist_digits_polariton_snn_dynamic Pipeline

This is an experimental application example outside the WF-/WJ- requirements.
The repository does not claim validated MNIST classification accuracy, and the
automated test verifies data flow and determinism rather than model quality.

`src/mnist_digits_polariton_snn_dynamic/` is a dynamic MNIST classification
workflow built on top of `polarism`. It downscales digits to a 7x7 input, maps
pixels to pulsed Gaussian pump energies, simulates condensate and reservoir
dynamics, and trains a logistic-regression readout.

Each 7x7 image intensity is mapped linearly to one of 49 pulse energies. The
readout receives time-resolved condensate-density values integrated over 49
spot masks plus one global mask; optional active and inactive reservoir traces
are appended, and `raw`, `summary`, or `both` selects their feature reduction.
A standardized multinomial logistic regression is fitted only on the
stratified training split and produces integer predictions for the held-out
split. The deterministic synthetic end-to-end test uses 20 generated inputs
and does not assert or publish an accuracy value.

The workflow keeps reusable physics configuration in:

```text
src/mnist_digits_polariton_snn_dynamic/polarism_base.yaml
```

Campaign scenarios live in:

```text
src/mnist_digits_polariton_snn_dynamic/scenarios/
```

## Deterministic Initial Conditions

The resolved Polarism config must set `physics.init_seed` explicitly. Both the
single-sample runner and the batched runner fail fast when the seed is missing
or `null`, so single-image debugging and batched production runs use equivalent
initial-condition semantics.

The current base config uses:

```yaml
physics:
  init_seed: 42
```

## Pitch/Sigma Sweep Gate

The pitch/sigma campaign checks whether changing pump spacing and spot width
affects spatial mixing. Validate the discrete Gaussian integral against the
analytic spatial integral before submitting the campaign:

```text
sum(exp(-0.5 * r^2 / sigma^2)) * dx * dy
2*pi*sigma^2
```

For the current `1024x1024`, `160 um x 160 um` grid, the accepted threshold is
`1.0e-4` relative error. The current sweep is at floating-point roundoff, with
worst observed lattice-center error `3.18e-16`; no hard sigma/dx validation or
scenario-specific grid refinement is required by this measurement.

If a future scenario exceeds the threshold, do not submit it directly. Either
add an explicit sigma/dx guard in `GridLatticeGeometry`, or refine the scenario
grid and account for the increased FFT cost.

## Threshold Calibration

The staged campaign first runs a GPU calibration job for each scenario. The
calibration uses a single pump spot at the scenario lattice center and sweeps
power between the scenario encoder bounds. A point is treated as condensed when
the peak center-channel condensate trace exceeds the first readout frame by the
configured ratio, currently `10`.

The first scenario, or the manifest's explicit `baseline_scenario_id`, defines
the baseline threshold multiplier:

```text
baseline encoding.power_max / baseline calibration.threshold_power
```

Each full run then uses:

```text
scenario calibration.threshold_power * baseline threshold multiplier
```

This keeps variants at a comparable distance from their own single-spot
threshold without editing scenario YAML files by hand.

The condensation proxy is an explicit campaign assumption. Confirm it with a
pilot calibration before submitting the full Rysy sweep.

## Ablation Finalize

The full GPU run stores `input_images.npy`, `encoded_powers.npy`, labels, raw
trace arrays, and the original classifier features. The CPU-only finalize stage
recomputes readout matrices from raw traces and evaluates:

- flattened pixel baseline
- full summary plus raw trace features
- one field and one statistic at a time
- all statistics for one field
- one statistic across all recorded fields

Each subset reports held-out accuracy and `delta_vs_pixel_baseline`. Scenario
reports are written to `ablation_report.json`; the campaign-level comparison is
written to `campaign_ablation_summary.json`.

Those values belong to the explicitly configured external campaign that
created its output directory. They are not reproduced by the synthetic flow
test and are not evidence of validated MNIST accuracy.

## Slurm Chain

The campaign reads the shared repository-root scheduler configuration. Create
it before the first submission with `cp slurm.env.example slurm.env`, then set
the Slurm account, resources, persistent run directory, and any `SNN_MIX_*`
overrides required by the campaign.

`cluster/submit_campaign.sh` submits three dependent jobs per scenario:

```text
<id>_calibrate -> <id>_run -> <id>_finalize
```

Scenario chains are limited by `MAX_CONCURRENT_SCENARIOS` from `slurm.env`.
With the current value `2`, two chains may progress at once. Run stages depend
on their own calibration and on baseline calibration. Finalize stages are
CPU-only.

Use dry-run first:

```bash
bash src/mnist_digits_polariton_snn_dynamic/cluster/submit_campaign.sh --dry-run src/mnist_digits_polariton_snn_dynamic/scenarios/pitch_sigma_sweep/manifest.yaml
```

Do not submit the real queue until the calibration proxy is accepted and one
full baseline GPU timing pilot has been measured on Rysy.

## Local Diagnostics

Campaign-specific diagnostics and profiling tools are intentionally local and
are excluded from version control. They do not form part of the workflow
contract.
