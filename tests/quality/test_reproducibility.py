from __future__ import annotations

import h5py
import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def test_seed_reproducibility() -> None:
    compute_engine.xp = np
    first = SimulationController(small_config(physics__init_seed=42)); first.run()
    second = SimulationController(small_config(physics__init_seed=42)); second.run()
    third = SimulationController(small_config(physics__init_seed=43)); third.run()
    assert np.array_equal(first.state.psi, second.state.psi)
    assert not np.array_equal(first.state.psi, third.state.psi)


def test_hdf5_persists_configuration_metadata(tmp_path) -> None:
    compute_engine.xp = np
    controller = SimulationController(small_config(result__save_results=True, result__save_hdf5=True, result__output_directory=str(tmp_path)))
    controller.run()
    with h5py.File(controller.storage_visitor.output_dir / "results.h5") as result:
        assert "config" in result
        assert "json" in result["config"].attrs
