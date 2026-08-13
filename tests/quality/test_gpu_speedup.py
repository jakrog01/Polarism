from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.env_metadata import collect_env_metadata, write_hardware_tex
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


@pytest.mark.slow
@pytest.mark.gpu
def test_ifrk4_gpu_speedup_artifact() -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    if not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA device unavailable")
    entries = []
    for nx in (128, 256, 512, 1024):
        times = []
        for use_gpu in (False, True):
            cfg = small_config(grid__nx=nx, grid__ny=nx, solver__method="ifrk4-fft-cuda", solver__total_time=0.1, compute_engine=ComputeEngineParameters(use_gpu=use_gpu))
            start = time.perf_counter(); SimulationController(cfg).run(); times.append(time.perf_counter() - start)
        entries.append({"nx": nx, "cpu_time_s": times[0], "gpu_time_s": times[1]})
    Path("artifacts/benchmark").mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path("artifacts/benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts/benchmark/gpu_speedup.json").write_text(
        json.dumps({"environment": collect_env_metadata(), "entries": entries})
    )
    write_hardware_tex(benchmark_dir / "hardware.tex")
    assert entries[-1]["gpu_time_s"] < entries[-1]["cpu_time_s"]
