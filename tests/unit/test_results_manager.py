from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.results.result_node import ResultNode
from polarism.results.results_manager import ResultsManager
from polarism.results.visitors.result_visitor import ResultVisitor


class Visitor(ResultVisitor):
    def __init__(self, names, location="cpu", fail=False, fatal=True):
        self.names, self.data_location, self.fail, self.fatal_on_error = set(names), location, fail, fatal
        self.calls = 0
        self.cached = None
        self.abort = Mock()

    def needs(self, nodes):
        return self.names

    def visit_all(self, nodes, **context):
        self.calls += 1
        self.cached = context["cached"]
        if self.fail:
            raise RuntimeError("visitor failure")


def _node(counter, reduce):
    return ResultNode("X", lambda **ctx: counter.__setitem__("n", counter["n"] + 1) or np.ones((2, 2)), reduce, None, None, None, True, True, None, True)


def test_results_manager_cache_and_optional_error_handling() -> None:
    compute_engine.xp = np
    counter = {"n": 0}; manager = ResultsManager(); manager.nodes = [_node(counter, lambda x: x.sum())]
    manager.add_visitor(Visitor({"X"})); manager.add_visitor(Visitor({"X"})); manager.step(0.0)
    assert counter["n"] == 1
    failing = Visitor({"X"}, fail=True, fatal=False)
    manager = ResultsManager(); manager.nodes = [_node({"n": 0}, lambda x: x.sum())]
    manager.add_visitor(failing); manager.add_visitor(Visitor({"X"})); manager.step(0.0); manager.step(1.0)
    assert failing.abort.call_count == 1 and id(failing) in manager._disabled_visitor_ids and failing.calls == 1


def test_results_manager_fatal_and_device_paths() -> None:
    compute_engine.xp = np
    reduce = Mock(return_value=1.0); manager = ResultsManager(); manager.nodes = [_node({"n": 0}, reduce)]
    device = Visitor({"X"}, "device"); manager.add_visitor(device); manager.step(0.0)
    reduce.assert_not_called(); assert device.cached["X"][1] is None
    reduce = Mock(return_value=1.0); manager = ResultsManager(); manager.nodes = [_node({"n": 0}, reduce)]
    device = Visitor({"X"}, "device"); manager.add_visitor(device); manager.add_visitor(Visitor({"X"})); manager.step(0.0)
    assert reduce.call_count == 1 and device.cached["X"][1] == 1.0
    manager = ResultsManager(); manager.nodes = [_node({"n": 0}, lambda x: x.sum())]; manager.add_visitor(Visitor({"X"}, fail=True))
    with pytest.raises(RuntimeError, match="visitor failure"):
        manager.step(0.0)
