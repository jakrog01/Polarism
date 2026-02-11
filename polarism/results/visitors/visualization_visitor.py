from __future__ import annotations

from polarism.results.real_time_visualization import RealTimeVisualization
from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor


class VisualizationVisitor(ResultVisitor):
    visualizer: RealTimeVisualization

    def __init__(self, visualizer: RealTimeVisualization):
        self.visualizer = visualizer

    def visit_all(self, nodes: list[ResultNode], **context):
        fields = {}
        scalars = {}
        scalar_groups = {}

        for node in nodes:
            field, reduced = node.compute_cpu(**context)
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
