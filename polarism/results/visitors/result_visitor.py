"""Base result visitor interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ResultVisitor(ABC):
    """Define the interface for result visitors."""

    data_location: str = "cpu"
    fatal_on_error: bool = True

    def needs(self, nodes: list) -> set[str]:
        """Return the set of node names this visitor requires.

        Override in subclasses to avoid computing unused nodes.
        """
        return {n.name for n in nodes}

    @abstractmethod
    def visit_all(self, nodes: list, **context) -> None:
        """Visit all result nodes."""
        raise NotImplementedError
