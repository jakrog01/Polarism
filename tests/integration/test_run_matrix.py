from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from polarism.solver.solver_compatibility import _SOLVER_CAPABILITIES
from tests.unit.conftest import small_config

TOL_INTEG_FINITE = 1e-12  # No NaN/Inf; buffer arithmetic identity.
CASES = [(solver, grid_type, reservoir) for solver, caps in _SOLVER_CAPABILITIES.items() for grid_type in caps["grid_types"] for reservoir in caps["reservoir_types"]]


@pytest.mark.parametrize("solver,grid_type,reservoir", CASES)
def test_solver_run_matrix(tmp_path, solver: str, grid_type: str, reservoir: str) -> None:
    requires_gpu = solver in {"rk4-cuda", "rk4-cuda-v100", "ifrk4-fft-cuda"}
    compute_engine.configure(ComputeEngineParameters(use_gpu=requires_gpu))
    cfg = small_config(grid__nx=32, grid__ny=32, grid__lx=40.0, grid__ly=40.0, grid__grid_type=grid_type, solver__method=solver, solver__dt=0.01, solver__total_time=0.5, reservoir__reservoir_type=reservoir, result__save_results=True, result__save_hdf5=True, result__save_interval=7, result__batch_size=10, result__output_directory=str(tmp_path), compute_engine=ComputeEngineParameters(use_gpu=requires_gpu))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        controller = SimulationController(cfg)
        controller.run()
    assert np.isfinite(compute_engine.to_cpu(controller.state.psi)).all()
    assert np.isfinite(compute_engine.to_cpu(controller.reservoir.get_reservoir_density())).all()
    assert int(controller.state.t / cfg.solver.dt) == int(cfg.solver.total_time / cfg.solver.dt)
    assert controller.storage_visitor is not None
    with __import__("h5py").File(controller.storage_visitor.output_dir / "results.h5") as result:
        assert result["time"].shape[0] == math.ceil(int(cfg.solver.total_time / cfg.solver.dt) / cfg.result.save_interval)
