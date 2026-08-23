# Local create_characteristic scenarios

Store local two-dimensional threshold-crossing sweep configurations here.
Files matching `*.yaml` in this directory are ignored by Git.

Use the naming pattern:

```text
gaas_<description>_characteristic.yaml
```

For example:

```text
gaas_9pulse_2um_characteristic.yaml
```

Run a local scenario with:

```bash
bash src/create_characteristic/submit.sh \
  --config src/create_characteristic/scenarios/gaas_9pulse_2um_characteristic.yaml
```

The scenario must define the two sweep axes (`energy_*` and `separation_*`) and
an `output.threshold_criterion`. Set `output.save_per_point_trace: true` to
write a marked time-trace plot for every grid point.
