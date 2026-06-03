"""Tests for polariton_hpc_pipeline parameter sweep expansion."""
from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src" / "polariton_hpc_pipeline"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

from pipeline.config.sweep import expand_parameter_sweep


def _base_cfg(
    laser_power: float | str,
    power_modifiers: list[dict] | None = None,
) -> dict:
    return {
        "global": {
            "grid": {"lx": 160.0, "ly": 160.0},
            "laser_defaults": {
                "laser_type": "pulse-gaussian",
                "power_definition": "pulse_energy",
                "sigma_space": 2.0,
                "sigma_time": 1.7,
                "cutoff_sigma": 3.0,
            },
            "parameter_sweep": {
                "enabled": True,
                "power_values": [100.0],
                "sigma_time_values": [1.7],
                "pulse_separation_values": [12.0],
                "sigma_space_values": [2.0],
            },
        },
        "scenarios": [
            {
                "name": "spot",
                "timing_vars": {"cycle_duration": "pulse_separation"},
                "power_modifiers": power_modifiers or [],
                "lasers": [
                    {
                        "id": "center",
                        "power": laser_power,
                        "x0": 0.0,
                        "y0": 0.0,
                        "delay": 0.0,
                        "pulse_separation": "cycle_duration",
                        "n_pulses": 3,
                    }
                ],
            }
        ],
    }


def test_sweep_label_uses_absolute_laser_power_when_config_is_absolute() -> None:
    """Absolute scenario powers should appear in names and sweep metadata."""
    cfg = _base_cfg(3500.0)

    expanded, names, _threshold = expand_parameter_sweep(cfg)

    assert names == ["spot_E3500_sep12"]
    scenario = expanded["scenarios"][0]
    assert scenario["sweep"]["power"] == 3500.0
    assert scenario["lasers"][0]["power"] == 3500.0


def test_sweep_label_keeps_reference_power_when_config_is_p_relative() -> None:
    """P-relative powers should keep the swept reference value in labels."""
    cfg = _base_cfg("0.6P")

    expanded, names, _threshold = expand_parameter_sweep(cfg)

    assert names == ["spot_E100_sep12"]
    scenario = expanded["scenarios"][0]
    assert scenario["sweep"]["power"] == 100.0
    assert scenario["lasers"][0]["power"] == 60.0


def test_sweep_label_treats_p_relative_power_modifier_as_reference_power() -> None:
    """P-relative power modifiers are still reference-power sweeps."""
    cfg = _base_cfg("P", [{"ids": ["center"], "power": "0.6P"}])

    expanded, names, _threshold = expand_parameter_sweep(cfg)

    assert names == ["spot_E100_sep12"]
    scenario = expanded["scenarios"][0]
    assert scenario["sweep"]["power"] == 100.0
    assert scenario["lasers"][0]["power"] == 60.0
