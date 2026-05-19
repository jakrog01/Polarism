# Local scenario configs

This directory is for local `pump_multi_comparison` campaign YAML files.

Tracked repository code keeps `../config.yaml` as the default runnable example.
Ad-hoc validation, production, and exploratory campaign configs should live here
as local files and are ignored by Git through:

```gitignore
src/pump_multi_comparison/scenarios/*.yaml
```

Run them explicitly:

```bash
cd src/pump_multi_comparison
bash submit.sh --config scenarios/<campaign>.yaml --dry-run
bash submit.sh --config scenarios/<campaign>.yaml
```

