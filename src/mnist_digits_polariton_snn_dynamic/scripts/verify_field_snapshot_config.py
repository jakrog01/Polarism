from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_digits_polariton_snn_dynamic.config.loader import (
    FieldSnapshotConfig,
    load_snn_dynamic_config,
)


cfg_disabled = FieldSnapshotConfig()
assert cfg_disabled.enabled is False
assert cfg_disabled.sample_indices == ()

raw = {
    "polarism_config_path": "polarism_base.yaml",
    "data": {"path": "mnist.npz", "n_samples": 10, "seed": 0},
    "geometry": {
        "center_x_um": 0.0,
        "center_y_um": 0.0,
        "pitch_um": 12.0,
        "sigma_space_um": 2.0,
    },
    "pulse": {
        "sigma_time": 1.5,
        "pulse_separation": 10.0,
        "n_pulses": 10,
        "cutoff_sigma": 3.0,
        "power_definition": "peak",
    },
    "encoding": {"power_min": 0.0, "power_max": 800.0},
    "readout": {
        "warmup_ps": 0.0,
        "stride_steps": 1,
        "mask_radius_um": 3.0,
        "record_reservoir": False,
        "feature_mode": "raw",
    },
    "classifier": {"test_fraction": 0.2, "C": 1.0, "max_iter": 100},
    "output_dir": "out",
    "field_snapshots": {
        "enabled": True,
        "sample_indices": [0, 2],
        "times_ps": [1.0, 2.0],
        "downsample_factor": 2,
        "include_pump": True,
    },
}

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_snn_dynamic_config(str(path))

assert loaded.field_snapshots == FieldSnapshotConfig(
    enabled=True,
    sample_indices=(0, 2),
    times_ps=(1.0, 2.0),
    downsample_factor=2,
    include_pump=True,
)

print("OK: default and YAML field snapshot configuration")
