"""Tests for polariton_hpc_pipeline parameter sweep expansion."""
from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_SRC = Path(__file__).resolve().parents[1] / "src" / "polariton_hpc_pipeline"
if str(PIPELINE_SRC) not in sys.path:
    sys.path.insert(0, str(PIPELINE_SRC))

THRESHOLD_SRC = Path(__file__).resolve().parents[1] / "src" / "threshold_finder"
if str(THRESHOLD_SRC) not in sys.path:
    sys.path.insert(0, str(THRESHOLD_SRC))

from pipeline.config.sweep import expand_parameter_sweep
from threshold_finder.config.loader import generate_sweep_points


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


def test_sweep_expands_all_requested_axes() -> None:
    cfg = _base_cfg("P")
    sweep = cfg["global"]["parameter_sweep"]
    sweep["power_values"] = [1.0, 2.0, 3.0]
    sweep["sigma_time_values"] = [1.0, 2.0]
    sweep["pulse_separation_values"] = [3.0, 4.0, 5.0, 6.0]
    sweep["sigma_space_values"] = [2.0]
    expanded, names, _ = expand_parameter_sweep(cfg)
    assert len(expanded["scenarios"]) == 24 and len(set(names)) == 24
    assert all("sweep" in scenario for scenario in expanded["scenarios"])


def test_threshold_sweep_seed_expansion_preserves_default_and_orders_ensembles() -> None:
    cfg = {"sweep": {"P_min": 0.1, "P_max": 0.4, "P_step": 0.1}}
    default = generate_sweep_points(cfg)
    assert default == [
        {"index": index, "P": power, "sweep_variable": "P", "sweep_value": power}
        for index, power in enumerate((0.1, 0.2, 0.3, 0.4))
    ]
    seeded = generate_sweep_points({"sweep": {**cfg["sweep"], "seeds": [1, 2, 3]}})
    assert len(seeded) == 12
    assert [point["index"] for point in seeded] == list(range(12))
    assert [(point["P"], point["seed"]) for point in seeded] == [
        (power, seed) for power in (0.1, 0.2, 0.3, 0.4) for seed in (1, 2, 3)
    ]
