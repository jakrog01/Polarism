from abc import ABC, abstractmethod

from polarism.results.result_node import ResultNode


class ResultProvider(ABC):
    @abstractmethod
    def make_result_nodes(self) -> list["ResultNode"]:
        raise NotImplementedError
