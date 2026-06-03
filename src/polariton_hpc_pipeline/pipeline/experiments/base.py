"""ExperimentSpec protocol — the interface every experiment module must satisfy."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExperimentSpec(Protocol):
    """Experiment-specific extension points for the shared HPC pipeline."""

    name: str

    def matches(self, cfg: dict[str, Any]) -> bool:
        """Return True if this experiment owns the given config."""
        ...

    def validate(self, cfg: dict[str, Any]) -> list[str]:
        """Return experiment-specific validation errors (empty list = valid)."""
        ...

    def expand_parameter_sweep(
        self, cfg: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
        """Expand scenarios. Returns (expanded_cfg, names, threshold_stub)."""
        ...

    def build_calibration_scenarios(
        self, cfg: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        """Return named calibration scenario dicts (None = phase disabled)."""
        ...

    def summarize(
        self, scenarios: list[str], run_dir: str, results_dir: str
    ) -> bool:
        """Generate experiment-specific summaries. Return True if handled."""
        ...
