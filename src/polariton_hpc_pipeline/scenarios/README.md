# Local scenario configs

This directory is for local `polariton_hpc_pipeline` campaign YAML files.

Tracked repository code keeps `../config.yaml` as the default runnable example.
Ad-hoc validation, production, and exploratory campaign configs should live here
as local files and are ignored by Git through:

```gitignore
src/polariton_hpc_pipeline/scenarios/*.yaml
```

Run them explicitly:

```bash
cd src/polariton_hpc_pipeline
bash submit.sh --config scenarios/<campaign>.yaml --dry-run
bash submit.sh --config scenarios/<campaign>.yaml
```

