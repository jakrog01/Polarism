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

TIMED_REPETITIONS = 7
BENCHMARK_TOTAL_TIME_S = 0.1


def _run_timed(cfg, use_gpu: bool) -> float:
    start = time.perf_counter()
    SimulationController(cfg).run()
    if use_gpu:
        compute_engine.xp.cuda.Device().synchronize()
    return time.perf_counter() - start


def _statistics(samples: list[float]) -> dict[str, float | list[float]]:
    values = np.asarray(samples, dtype=np.float64)
    q25, q75 = np.quantile(values, (0.25, 0.75))
    return {
        "samples_s": samples,
        "median_s": float(np.median(values)),
        "min_s": float(np.min(values)),
        "iqr_s": float(q75 - q25),
    }


@pytest.mark.slow
@pytest.mark.gpu
def test_ifrk4_gpu_speedup_artifact() -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    if not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA device unavailable")
    entries: list[dict[str, float | int | list[float]]] = []
    for nx in (128, 256, 512, 1024):
        measurements: dict[bool, dict[str, float | list[float]]] = {}
        for use_gpu in (False, True):
            cfg = small_config(grid__nx=nx, grid__ny=nx, solver__method="ifrk4-fft-cuda", solver__total_time=BENCHMARK_TOTAL_TIME_S, compute_engine=ComputeEngineParameters(use_gpu=use_gpu))
            SimulationController(cfg).run()
            if use_gpu:
                compute_engine.xp.cuda.Device().synchronize()
            samples = [_run_timed(cfg, use_gpu) for _ in range(TIMED_REPETITIONS)]
            measurements[use_gpu] = _statistics(samples)
        cpu = measurements[False]
        gpu = measurements[True]
        entries.append({
            "nx": nx,
            "cpu_time_s": cpu["median_s"],
            "gpu_time_s": gpu["median_s"],
            "cpu_samples_s": cpu["samples_s"],
            "gpu_samples_s": gpu["samples_s"],
            "cpu_median_s": cpu["median_s"],
            "gpu_median_s": gpu["median_s"],
            "cpu_min_s": cpu["min_s"],
            "gpu_min_s": gpu["min_s"],
            "cpu_iqr_s": cpu["iqr_s"],
            "gpu_iqr_s": gpu["iqr_s"],
            "speedup": float(cpu["median_s"]) / float(gpu["median_s"]),
        })
    benchmark_dir = Path("artifacts/benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts/benchmark/gpu_speedup.json").write_text(
        json.dumps(
            {
                "environment": collect_env_metadata(),
                "measurement": {
                    "warmup_repetitions": 1,
                    "timed_repetitions": TIMED_REPETITIONS,
                    "solver_total_time_s": BENCHMARK_TOTAL_TIME_S,
                    "statistics": ["median_s", "min_s", "iqr_s"],
                },
                "entries": entries,
            }
        )
    )
    write_hardware_tex(benchmark_dir / "hardware.tex")
    assert entries[-1]["gpu_time_s"] < entries[-1]["cpu_time_s"]
