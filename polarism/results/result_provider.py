"""Interfaces for objects that expose results."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.results.result_node import ResultNode


class ResultProvider(ABC):
    """Define the interface for result providers."""
    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes exposed by this object."""
        raise NotImplementedError
