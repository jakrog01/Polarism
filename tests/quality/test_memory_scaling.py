from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config

MEMORY_BUDGET_BYTES = 5 * 1024 * 1024  # HDF5 batches should avoid sample-count-scaled buffers.


@pytest.mark.slow
def test_storage_memory_scaling(tmp_path) -> None:
    compute_engine.xp = np
    peaks = []
    for interval in (10, 1):
        cfg = small_config(result__save_results=True, result__save_hdf5=True, result__output_directory=str(tmp_path / str(interval)), result__save_interval=interval, result__batch_size=10, solver__total_time=0.2)
        tracemalloc.start(); SimulationController(cfg).run(); peaks.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
    Path("artifacts/benchmark").mkdir(parents=True, exist_ok=True)
    Path("artifacts/benchmark/memory_scaling.json").write_text(json.dumps({"sparse_peak": peaks[0], "dense_peak": peaks[1]}))
    assert peaks[1] - peaks[0] < MEMORY_BUDGET_BYTES
