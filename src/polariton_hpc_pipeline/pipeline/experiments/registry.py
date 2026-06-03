"""Experiment registry: maps configs to ExperimentSpec instances."""
from __future__ import annotations

from typing import Any

from pipeline.experiments.base import ExperimentSpec

_experiments: list[ExperimentSpec] = []


def register(experiment: ExperimentSpec) -> None:
    _experiments.append(experiment)


def get_experiment(cfg: dict[str, Any]) -> ExperimentSpec:
    for exp in _experiments:
        if exp.matches(cfg):
            return exp
    from pipeline.experiments.generic import GenericExperiment
    return GenericExperiment()


def _bootstrap() -> None:
    from pipeline.experiments.probe5_trap_gate import Probe5TrapGate
    register(Probe5TrapGate())


_bootstrap()
