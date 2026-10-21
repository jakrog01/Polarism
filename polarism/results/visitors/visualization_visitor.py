"""Result visualization visitors."""
from __future__ import annotations

from polarism.results.real_time_visualization import RealTimeVisualization
from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor


class VisualizationVisitor(ResultVisitor):
    """Send result nodes to the live visualizer."""
    visualizer: RealTimeVisualization

    def __init__(self, visualizer: RealTimeVisualization, member: int | None = None):
        """Store the live visualizer."""
        self.visualizer = visualizer
        self.member = member

    def needs(self, nodes: list[ResultNode]) -> set[str]:
        """Return names of nodes that are exposed for visualization."""
        return {n.name for n in nodes if n.expose}

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
            if self.member is not None:
                field = field[self.member] if getattr(field, "ndim", 0) else field
                reduced = reduced[self.member] if getattr(reduced, "ndim", 0) else reduced
            if getattr(node, "cmap", None) is not None:
                fields[node.name] = field
            else:
                scalars[node.name] = float(reduced)
        if "scalar_groups" in context and context["scalar_groups"]:
            scalar_groups = {
                group: {label: float(value[self.member] if self.member is not None else value)
                        for label, value in values.items()}
                for group, values in context["scalar_groups"].items()
            }

        self.visualizer.update(
            t=context.get("t", 0.0),
            fields=fields,
            scalars=scalars,
            scalar_groups=scalar_groups,
        )
