from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_SRC = Path(__file__).resolve().parents[2] / "src" / "polariton_hpc_pipeline"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from pipeline.config.sweep import expand_parameter_sweep


def test_parameter_sweep_cartesian_expansion_and_metadata() -> None:
    cfg = {
        "global": {
            "grid": {"lx": 40.0, "ly": 40.0},
            "laser_defaults": {"laser_type": "pulse-gaussian", "power_definition": "peak_amplitude", "sigma_time": 1.0, "sigma_space": 2.0},
            "parameter_sweep": {"enabled": True, "power_values": [1.0, 2.0, 3.0], "sigma_time_values": [0.5, 1.0], "pulse_separation_values": [2.0, 3.0, 4.0, 5.0], "sigma_space_values": [2.0]},
        },
        "scenarios": [{"name": "spot", "lasers": [{"id": "center", "power": "P", "n_pulses": 2}]}],
    }
    expanded, names, _ = expand_parameter_sweep(cfg)
    assert len(names) == 24 and len(set(names)) == 24
    for scenario in expanded["scenarios"]:
        assert scenario["sweep"]["power"] in {1.0, 2.0, 3.0}
        assert scenario["lasers"][0]["power"] == scenario["sweep"]["power"]
