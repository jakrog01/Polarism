from __future__ import annotations

from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor

class ResultsManager:
    def __init__(self):
        self.t: list[float] = []
        self.nodes: list[ResultNode] = []
        self.visitors: list[ResultVisitor] = []

    def add_visitor(self, visitor: ResultVisitor) -> None:
        self.visitors.append(visitor)

    def step(self, t: float, **context) -> None:
        self.t.append(t)
        for visitor in self.visitors:
            visitor.visit_all(self.nodes, t=t, **context)
