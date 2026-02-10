from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polarism.results.result_node import ResultNode


class ResultProvider(ABC):
    @abstractmethod
    def make_result_nodes(self) -> list[ResultNode]:
        raise NotImplementedError
