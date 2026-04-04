"""Result visualization visitors."""
from __future__ import annotations

from polarism.results.real_time_visualization import RealTimeVisualization
from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor


class VisualizationVisitor(ResultVisitor):
    """Send result nodes to the live visualizer."""
    visualizer: RealTimeVisualization

    def __init__(self, visualizer: RealTimeVisualization):
        """Store the live visualizer."""
        self.visualizer = visualizer

    def visit_all(self, nodes: list[ResultNode], **context):
        """Send one result step to the visualizer."""
        cached = context.get("cached", {})
        fields = {}
        scalars = {}
        scalar_groups = {}

        for node in nodes:
            if not node.expose:
                continue
            field, reduced = cached[node.name]
            if getattr(node, "cmap", None) is not None:
                fields[node.name] = field
            else:
                scalars[node.name] = reduced
        if "scalar_groups" in context and context["scalar_groups"]:
            scalar_groups = context["scalar_groups"]

        self.visualizer.update(
            t=context.get("t", 0.0),
            fields=fields,
            scalars=scalars,
            scalar_groups=scalar_groups,
        )
