# Campaign Scenarios

Scenario campaigns run one complete `SNNDynamicConfig` YAML per scenario. A campaign directory contains a `manifest.yaml` plus the scenario YAML files referenced by that manifest.

Each scenario YAML is self-contained and uses the same schema as `config.yaml`. The campaign runner ignores the scenario `output_dir` and writes results to `<campaign_output_dir>/<scenario_id>/`.

## Pre-Submission Gates

For `pitch_sigma_sweep`, run the CPU-only spatial discretization check before
submitting GPU jobs:

```bash
.venv/bin/python src/mnist_digits_polariton_snn_dynamic/scripts/check_pitch_sigma_discretization.py
```

The script evaluates every scenario Gaussian on the configured `1024x1024`,
`160 um x 160 um` periodic grid and compares the discrete integral against
`2*pi*sigma_space_um^2`. The current sweep is below the accepted `1.0e-4`
relative-error threshold by numerical roundoff, with worst observed lattice
error `3.18e-16`.

If a future pitch/sigma scenario fails this check, do not submit it as-is.
Choose one of two explicit fixes before GPU submission: add a sigma/dx
validation guard in `GridLatticeGeometry`, or refine the scenario grid and
account for the increased FFT cost.

## Campaign Stages

`cluster/submit_campaign.sh` submits three jobs per scenario:

- `<id>_calibrate`: GPU threshold calibration, writes `calibration.json`.
- `<id>_run`: GPU full run, depends on own calibration and writes raw traces
  and classifier outputs.
- `<id>_finalize`: CPU-only ablation, depends on the full run, writes
  `ablation_report.json`.

Each scenario calibration estimates `threshold_power_full_lattice`, applies a
central single-spot guard, and writes the scenario-local `final_power_max` used
by the full run. The manifest `baseline_scenario_id` remains the reporting
reference only.

The submitter runs scenarios sequentially. Later scenario calibrations depend on
the finalize job from the immediately preceding scenario.

Submit a campaign from the package directory with:

```bash
bash cluster/submit_campaign.sh --dry-run scenarios/pitch_sigma_sweep/manifest.yaml
```

Submit without `--dry-run` only after the condensation proxy is accepted.
