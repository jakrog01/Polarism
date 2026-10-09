from __future__ import annotations

import subprocess
import sys

import h5py
import matplotlib
import numpy as np

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_cli_help() -> None:
    completed = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0
    assert any(token in completed.stdout for token in ("Usage:", "usage:", "--help"))


def test_programmatic_hdf5_output_and_figure(tmp_path) -> None:
    compute_engine.xp = np
    cfg = small_config(result__save_results=True, result__save_hdf5=True, result__output_directory=str(tmp_path), result__save_interval=1, result__batch_size=2)
    controller = SimulationController(cfg); controller.run()
    assert controller.storage_visitor is not None
    output = controller.storage_visitor.output_dir / "results.h5"
    with h5py.File(output) as result:
        time = result["time"][:]
        norm = result["scalars/N(t)"][:]
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(time, norm); figure = tmp_path / "fig.png"; plt.savefig(figure); plt.close()
    assert figure.is_file() and figure.stat().st_size > 0
