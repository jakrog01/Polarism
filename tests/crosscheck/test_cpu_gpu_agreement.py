from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import compute_engine
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests._helpers import rel_l2_err
from tests.unit.conftest import small_config

TOL_CPU_GPU_F64 = 1e-10  # CPU/GPU double-precision relative L2 threshold.
TOL_CPU_GPU_F32 = 1e-5  # CPU/GPU single-precision relative L2 threshold.


@pytest.mark.gpu
@pytest.mark.parametrize("precision,tolerance", [("double", TOL_CPU_GPU_F64), ("single", TOL_CPU_GPU_F32)])
def test_cpu_gpu_solution_agreement(precision: str, tolerance: float) -> None:
    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    if not hasattr(compute_engine.xp, "cuda"):
        pytest.skip("CUDA device unavailable")
    gpu_cfg = small_config(solver__precision=precision, compute_engine=ComputeEngineParameters(use_gpu=True))
    gpu = SimulationController(gpu_cfg)
    initial = np.random.default_rng(0).standard_normal(gpu.state.psi.shape) + 1j * np.random.default_rng(1).standard_normal(gpu.state.psi.shape)
    gpu.state.psi[...] = compute_engine.xp.asarray(initial, dtype=gpu.state.psi.dtype)
    gpu.run(); gpu_psi = compute_engine.to_cpu(gpu.state.psi)
    cpu_cfg = small_config(solver__precision=precision, compute_engine=ComputeEngineParameters(use_gpu=False))
    cpu = SimulationController(cpu_cfg); cpu.state.psi[...] = initial.astype(cpu.state.psi.dtype); cpu.run()
    measured = rel_l2_err(gpu_psi, np.asarray(cpu.state.psi))
    assert measured < tolerance
