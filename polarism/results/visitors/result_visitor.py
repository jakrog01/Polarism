"""Base result visitor interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ResultVisitor(ABC):
    """Define the interface for result visitors."""
    @abstractmethod
    def visit_all(self, nodes: list, **context) -> None:
        """Visit all result nodes."""
        raise NotImplementedError
