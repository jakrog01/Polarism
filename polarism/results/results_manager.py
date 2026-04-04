"""Result visitor management."""
from __future__ import annotations

from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor


class ResultsManager:
    """Send result nodes to each visitor."""
    def __init__(self):
        """Set up empty result nodes and visitors."""
        self.nodes: list[ResultNode] = []
        self.visitors: list[ResultVisitor] = []

    def add_visitor(self, visitor: ResultVisitor) -> None:
        """Add one result visitor."""
        self.visitors.append(visitor)

    def step(self, t: float, **context) -> None:
        """Send one time step to each visitor."""
        cached = {}
        for node in self.nodes:
            field_cpu, reduced_cpu = node.compute_cpu(**context)
            cached[node.name] = (field_cpu, reduced_cpu)

        for visitor in self.visitors:
            visitor.visit_all(self.nodes, t=t, cached=cached, **context)
