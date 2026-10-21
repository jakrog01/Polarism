"""Batched benchmark timing and reporting."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from polarism import benchmark


def test_batch_mode_warms_both_paths_with_timed_step_count(monkeypatch, capsys):
    calls = []

    def controller(method, size, precision, reservoir, tableau="rk4", batch=1):
        solver = SimpleNamespace(advance=lambda *args: calls.append((batch, args[-2:])))
        return SimpleNamespace(solver=solver, potential=None, pump_field=None,
                               reservoir=None, boundary_condition=None, state=None)

    ticks = iter(range(6))
    monkeypatch.setattr(benchmark, "_controller", controller)
    monkeypatch.setattr(benchmark, "_copy_bandwidth", lambda cp: 1.0)
    monkeypatch.setattr(benchmark, "has_cuda_device", lambda: True)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    device = SimpleNamespace(synchronize=lambda: None)
    runtime = SimpleNamespace(getDevice=lambda: 0,
                              getDeviceProperties=lambda index: {"name": b"Test GPU", "major": 8, "minor": 9},
                              runtimeGetVersion=lambda: 12000, driverGetVersion=lambda: 12000)
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime, Device=lambda: device), __version__="test")
    monkeypatch.setitem(__import__("sys").modules, "cupy", cp)

    assert benchmark.main(["--methods", "rk4-cuda", "--grid-sizes", "4",
                           "--precisions", "single", "--steps", "7", "--batch", "2"]) == 0
    assert calls == [(2, (0, 7)), (2, (7, 7)),
                     (1, (0, 7)), (1, (7, 7)),
                     (1, (0, 7)), (1, (7, 7))]
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["speedup"] == 2
    assert record["ns_per_member_element_step"] == pytest.approx(1e9 / (2 * 4 * 4 * 7))


def test_batch_rejects_single_member():
    with pytest.raises(SystemExit, match="2"):
        benchmark.main(["--batch", "1"])
