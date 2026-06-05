# mnist_spacetime_v1

Scenario-based MNIST-branded mechanism pipeline for polariton spacetime/ballistic
experiments. This is intentionally separate from `polariton_hpc_pipeline` and
does not implement a classifier yet. The goal is to probe geometries that can
convert written reservoir patterns and ballistic timing into measurable spatial
or ROI-trace responses.

## Architectures

- `dynamic_hologram`: feature spots write a reservoir landscape, then a delayed
  read probe scatters through it. Output ROIs measure directional response.
- `ballistic_correlator`: input spots launch packets toward a mixer region,
  optionally with a central gate. Output ROIs measure coincidence and delay
  sensitivity.

## Rysy usage

Default dynamic-hologram pilot:

```bash
bash src/mnist_spacetime_v1/submit.sh \
  --config src/mnist_spacetime_v1/config.yaml
```

Ballistic correlator pilot:

```bash
bash src/mnist_spacetime_v1/submit.sh \
  --config src/mnist_spacetime_v1/scenarios/ballistic_correlator.yaml
```

Dry-run is access-node safe and does not execute Python:

```bash
bash src/mnist_spacetime_v1/submit.sh \
  --config src/mnist_spacetime_v1/config.yaml \
  --dry-run
```

## Outputs

Each run writes:

- `scenario_index.json` and `scenarios_expanded.json`
- `traces/<scenario>.npz`
- `metadata/<scenario>.json`
- `campaign_gpu_summary.json`
- `results_summary_spacetime.json`
- `summary_table.csv`
- `plots/<scenario>/pump_layout.png`
- `plots/<scenario>/roi_traces.png`
- `plots/<scenario>/scalar_traces.png`
- `plots/<scenario>/final_fields_downsampled.png`

The important diagnostic is not accuracy. Inspect ROI traces, output winner
stability, output margins, condensation timing, and whether delayed/symmetric
controls produce distinguishable responses.
