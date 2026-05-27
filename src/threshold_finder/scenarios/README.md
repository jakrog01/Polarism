# Local scenario configs

This directory is for local `threshold_finder` YAML files.

Tracked repository code keeps `../config.yaml` as the default GaAs runnable config.
Ad-hoc threshold scans and production parameter sweeps should live here as local
files and are ignored by Git through:

```gitignore
src/threshold_finder/scenarios/*.yaml
```

Run them explicitly:

```bash
cd src/threshold_finder
bash submit.sh --config scenarios/<campaign>.yaml --dry-run
bash submit.sh --config scenarios/<campaign>.yaml
```
