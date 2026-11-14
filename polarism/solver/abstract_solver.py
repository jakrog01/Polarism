"""Solver interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from polarism.compute_engine import compute_engine


@dataclass(frozen=True)
class StepReport:
    """Outcome of a solver step."""

    accepted: bool = True
    error: float | None = None
    dt_next: float | None = None


ACCEPTED = StepReport()


_BUILTIN_RESERVOIRS = frozenset({"single", "double", "quadratic-double", "none"})


@dataclass(frozen=True)
class SolverCapabilities:
    """Features accepted by a solver implementation.

    Every field has a conservative default, so a solver declares only what
    differs. Set-valued fields accept any iterable and are stored as
    frozensets.
    """

    components: frozenset[int] = frozenset({1})
    ndim: frozenset[int] = frozenset({2})
    grid_types: frozenset[str] = frozenset({"periodic"})
    reservoirs: frozenset[str] = _BUILTIN_RESERVOIRS
    custom_reservoirs: str = "quantitative"
    potential: bool = True
    kinetic_relaxation: bool = False
    dispersion: frozenset[str] = frozenset({"parabolic"})
    terms: bool = True
    noise: frozenset[str] = frozenset({"none", "wigner"})
    adaptive: bool = False
    coherent_drive: bool = False
    time_dependent_potential: bool = False
    quantitative_reservoirs: frozenset[str] = _BUILTIN_RESERVOIRS
    description: str = ""

    def __post_init__(self):
        for name in ("components", "ndim", "grid_types", "reservoirs",
                     "dispersion", "noise", "quantitative_reservoirs"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))


DEFAULT_CAPABILITIES = SolverCapabilities(description="Plugin solver with undeclared capabilities.")


class AbstractSolver(ABC):
    """Define the interface for solver backends.

    A solver that injects the Wigner increment inside its own step, at the
    point its scheme prescribes, sets ``injects_noise = True``. The default
    ``advance`` then does not add the increment again after each step.
    """

    capabilities: ClassVar[SolverCapabilities] = DEFAULT_CAPABILITIES
    injects_noise: ClassVar[bool] = False

    def __init__(self, config, grid, model):
        """Set up common solver data."""
        self.config = config
        self.physics = config.physics
        self.batch_physics = model.physics
        self.xp = compute_engine.xp

    @abstractmethod
    def step(self, model, state, dt) -> StepReport:
        """Advance one step and update state time and step on acceptance."""
        raise NotImplementedError

    def advance(self, model, state, n_steps: int) -> None:
        """Advance several steps and apply noise after accepted steps."""
        self.batch_physics.check(state.field_k if state._field is None else state.field)
        for _ in range(n_steps):
            completed = state.step
            report = self.step(model, state, self.config.solver.dt)
            if report.accepted and model.noise is not None and not self.injects_noise:
                model.noise.apply_to_state(state, model.reservoir, completed)
