"""GenericExperiment — fallback for parameter sweeps without a specific mode."""
from __future__ import annotations

from typing import Any


class GenericExperiment:
    """Handles any config not claimed by a specialised experiment."""

    name = "generic"

    def matches(self, cfg: dict[str, Any]) -> bool:
        return True

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        return []

    def expand_parameter_sweep(
        self, cfg: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
        from pipeline.config.sweep import expand_generic
        return expand_generic(cfg)

    def build_calibration_scenarios(
        self, cfg: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        return {}

    def summarize(
        self, scenarios: list[str], run_dir: str, results_dir: str
    ) -> bool:
        return False
