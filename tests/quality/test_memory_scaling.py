from __future__ import annotations

import gc
import json
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.env_metadata import collect_env_metadata
from polarism.simulation_controller import SimulationController
from tests._helpers import PeakMemoryMonitor
from tests.unit.conftest import small_config

GRID_SIZE = 256
DT = 0.001
N_STEPS = 640
BATCH_SIZE = 10
SAMPLE_COUNTS = (10, 40, 160, 640)
SLOPE_FRACTION_OF_PAYLOAD = 0.25


@dataclass(frozen=True)
class MemoryScalingPoint:
    save_interval: int
    sample_count: int
    peak_rss_bytes: int | None
    peak_tracemalloc_bytes: int
    hdf5_file_bytes: int
    total_payload_bytes: int


def _measure_point(tmp_path: Path, sample_count: int) -> MemoryScalingPoint:
    save_interval = N_STEPS // sample_count
    output_directory = tmp_path / f"samples-{sample_count}"
    cfg = small_config(
        grid__nx=GRID_SIZE,
        grid__ny=GRID_SIZE,
        solver__method="rk4-fdm-fused",
        solver__dt=DT,
        solver__total_time=N_STEPS * DT,
        result__save_results=True,
        result__save_hdf5=True,
        result__save_json=False,
        result__save_npy=False,
        result__output_directory=str(output_directory),
        result__save_interval=save_interval,
        result__batch_size=BATCH_SIZE,
        reservoir__expose_results=False,
        laser__expose_results=False,
    )
    controller = SimulationController(cfg)
    zero_pump = np.zeros_like(controller.grid.X)
    controller._compute_total_pump = lambda time: zero_pump
    gc.collect()
    tracemalloc.start()
    with PeakMemoryMonitor() as monitor:
        controller.run()
    peak_tracemalloc_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    hdf5_path = next(output_directory.rglob("results.h5"))
    with h5py.File(hdf5_path) as handle:
        actual_sample_count = int(handle["time"].shape[0])
        total_payload_bytes = sum(
            int(dataset.size * dataset.dtype.itemsize)
            for dataset in handle["fields"].values()
        )
    return MemoryScalingPoint(
        save_interval=save_interval,
        sample_count=actual_sample_count,
        peak_rss_bytes=monitor.result().peak_rss_bytes,
        peak_tracemalloc_bytes=peak_tracemalloc_bytes,
        hdf5_file_bytes=hdf5_path.stat().st_size,
        total_payload_bytes=total_payload_bytes,
    )


@pytest.mark.slow
def test_storage_memory_scaling(tmp_path: Path) -> None:
    """Verify WJ-05: "Zapis wyników nie powinien wymagać utrzymywania całej trajektorii symulacji w pamięci operacyjnej. Weryfikacja: Pomiar maksymalnego zużycia pamięci podczas długiego przebiegu dla rosnącej liczby zapisywanych próbek oraz test buforowanego zapisu danych do pliku HDF5."

    One stored float64 density field on the 256x256 grid occupies 512 KiB, and
    the densest 640-sample HDF5 payload occupies 320 MiB. The allowance is 0.25
    of the measured stored payload per sample, or 128 KiB per sample. It covers
    fixed-size batches, h5py caches, and allocator variation while remaining
    four times smaller than retaining every saved field in memory. All points
    execute the same 640-step solver loop, so only the sampling interval varies.
    """
    compute_engine.xp = np
    compute_engine.use_gpu = False
    points = [_measure_point(tmp_path, count) for count in SAMPLE_COUNTS]
    payload_bytes_per_sample = max(
        point.total_payload_bytes // point.sample_count for point in points
    )
    counts = np.asarray([point.sample_count for point in points], dtype=np.float64)
    trace_peaks = np.asarray(
        [point.peak_tracemalloc_bytes for point in points], dtype=np.float64
    )
    rss_available = all(point.peak_rss_bytes is not None for point in points)
    rss_peaks = (
        np.asarray([point.peak_rss_bytes for point in points], dtype=np.float64)
        if rss_available
        else None
    )
    trace_slope = float(np.polyfit(counts, trace_peaks, 1)[0])
    rss_slope = (
        None if rss_peaks is None else float(np.polyfit(counts, rss_peaks, 1)[0])
    )
    slope_threshold = SLOPE_FRACTION_OF_PAYLOAD * payload_bytes_per_sample
    full_trajectory_bytes = points[-1].total_payload_bytes
    trace_growth = int(trace_peaks[-1] - trace_peaks[0])
    rss_growth = None if rss_peaks is None else int(rss_peaks[-1] - rss_peaks[0])
    passed = (
        trace_slope < slope_threshold
        and trace_growth < full_trajectory_bytes
        and (rss_slope is None or rss_slope < slope_threshold)
        and (rss_growth is None or rss_growth < full_trajectory_bytes)
    )
    artifact = {
        "configuration": {
            "grid": {"nx": GRID_SIZE, "ny": GRID_SIZE},
            "dt": DT,
            "total_time": N_STEPS * DT,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "payload_bytes_per_sample": payload_bytes_per_sample,
            "save_intervals": [point.save_interval for point in points],
        },
        "points": [asdict(point) for point in points],
        "peak_tracemalloc_slope_bytes_per_sample": trace_slope,
        "peak_rss_slope_bytes_per_sample": rss_slope,
        "slope_threshold_bytes_per_sample": slope_threshold,
        "peak_tracemalloc_growth_bytes": trace_growth,
        "peak_rss_growth_bytes": rss_growth,
        "full_trajectory_bytes": full_trajectory_bytes,
        "passed": passed,
        "environment": collect_env_metadata(),
    }
    destination = Path("artifacts/benchmark/memory_scaling.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    assert points[-1].total_payload_bytes > 16 * BATCH_SIZE * payload_bytes_per_sample
    assert trace_slope < slope_threshold
    assert trace_growth < full_trajectory_bytes
    if rss_slope is None or rss_growth is None:
        pytest.fail("process RSS sampling from /proc/self/status was unavailable")
    assert rss_slope < slope_threshold
    assert rss_growth < full_trajectory_bytes
