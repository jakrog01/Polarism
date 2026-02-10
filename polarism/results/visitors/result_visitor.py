from __future__ import annotations

from abc import ABC, abstractmethod


class ResultVisitor(ABC):
    @abstractmethod
    def visit_all(self, nodes: list, **context) -> None:
        raise NotImplementedError
