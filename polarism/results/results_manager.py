"""Result visitor management."""
from __future__ import annotations

from polarism.compute_engine import compute_engine
from polarism.results.result_node import ResultNode
from polarism.results.visitors.result_visitor import ResultVisitor


class ResultsManager:
    """Send result nodes to each registered visitor.

    Visitors that declare ``data_location = "device"`` receive backend-native
    arrays without a CPU conversion.  Each node is computed at most once per
    :meth:`step` call regardless of how many visitors request it.

    When a node is only requested by device visitors, the scalar reduction
    (``reduce_dim_fn``) is skipped so GPU stream synchronisation is avoided.
    Nodes also needed by CPU visitors always compute both field and reduction.

    Visitor exceptions are isolated: a visitor that raises is disabled for the
    rest of the run so that it cannot abort the simulation loop.  Visitors that
    expose an ``abort(exc)`` method have it called before being disabled.
    """

    def __init__(self):
        self.nodes: list[ResultNode] = []
        self.visitors: list[ResultVisitor] = []
        self._disabled_visitor_ids: set[int] = set()

    def add_visitor(self, visitor: ResultVisitor) -> None:
        """Register a result visitor."""
        self.visitors.append(visitor)

    def step(self, t: float, **context) -> None:
        """Compute needed nodes once and dispatch to all visitors."""
        active = [v for v in self.visitors if id(v) not in self._disabled_visitor_ids]

        needed_device: set[str] = set()
        needed_cpu: set[str] = set()
        for visitor in active:
            ns = visitor.needs(self.nodes)
            if getattr(visitor, "data_location", "cpu") == "device":
                needed_device |= ns
            else:
                needed_cpu |= ns

        needed_full = needed_cpu
        needed_field_only = needed_device - needed_cpu

        device_cache: dict[str, tuple] = {}

        for node in self.nodes:
            if node.name in needed_full:
                field_dev, reduced_dev = node.compute_device(**context)
                device_cache[node.name] = (field_dev, reduced_dev)
            elif node.name in needed_field_only:
                field_dev = node.compute_field_device(**context)
                device_cache[node.name] = (field_dev, None)

        cpu_cache: dict[str, tuple] = {}
        for name in needed_cpu:
            if name in device_cache:
                fd, rd = device_cache[name]
                cpu_cache[name] = (compute_engine.to_cpu(fd), compute_engine.to_cpu(rd))

        import logging as _logging
        _log = _logging.getLogger(__name__)

        for visitor in active:
            loc = getattr(visitor, "data_location", "cpu")
            cache = device_cache if loc == "device" else cpu_cache
            try:
                visitor.visit_all(self.nodes, t=t, cached=cache, **context)
            except Exception as exc:
                if getattr(visitor, "fatal_on_error", True):
                    raise
                _log.error(
                    "ResultsManager: optional visitor %s raised at t=%.3f; disabling it. Error: %s",
                    type(visitor).__name__, t, exc, exc_info=True,
                )
                abort = getattr(visitor, "abort", None)
                if callable(abort):
                    try:
                        abort(exc)
                    except Exception:
                        pass
                self._disabled_visitor_ids.add(id(visitor))
