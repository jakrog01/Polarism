# Campaign Scenarios

Scenario campaigns run one complete `SNNDynamicConfig` YAML per scenario. A campaign directory contains a `manifest.yaml` plus the scenario YAML files referenced by that manifest.

Each scenario YAML is self-contained and uses the same schema as `config.yaml`. The campaign runner ignores the scenario `output_dir` and writes results to `<campaign_output_dir>/<scenario_id>/`.

## Pre-Submission Gates

For `pitch_sigma_sweep`, validate the spatial discretization before submitting
GPU jobs. Evaluate every scenario Gaussian on the configured `1024x1024`,
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

- `<id>_threshold`: CPU analytic threshold scan, writes `spike_threshold.json`.
- `<id>_run`: GPU full run, depends on its own threshold and writes raw traces
  and classifier outputs.
- `<id>_finalize`: CPU-only ablation, depends on the full run, writes
  `ablation_report.json`.

The threshold stage maximizes pulse-separated gain/loss crossings and writes
the scenario-local `final_power_max` used by the full run. The manifest
`baseline_scenario_id` remains the reporting reference only. Pass
`--with-calibrate` to submit the legacy GPU calibration between threshold and
run.

The submitter runs scenarios sequentially. Later threshold scans depend on the
finalize job from the immediately preceding scenario.

Submit a campaign from the package directory with:

```bash
bash cluster/submit_campaign.sh --dry-run scenarios/pitch_sigma_sweep/manifest.yaml
```

The legacy calibration path can be inspected with:

```bash
bash cluster/submit_campaign.sh --dry-run --with-calibrate scenarios/pitch_sigma_sweep/manifest.yaml
```
