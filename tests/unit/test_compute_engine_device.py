"""CUDA device selection through ComputeEngineParameters.gpu_device."""
import pytest

from polarism.compute_engine import ComputeEngine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters

pytestmark = pytest.mark.gpu


def test_gpu_device_becomes_current():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    engine = ComputeEngine()
    last = cp.cuda.runtime.getDeviceCount() - 1
    engine.configure(ComputeEngineParameters(use_gpu=True, gpu_device=last))
    assert cp.cuda.runtime.getDevice() == last
    assert engine.xp.zeros(1).device.id == last
    engine.configure(ComputeEngineParameters(use_gpu=True, gpu_device=0))


def test_gpu_device_out_of_range_raises():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    count = cp.cuda.runtime.getDeviceCount()
    with pytest.raises(ValueError, match="gpu_device"):
        ComputeEngine().configure(ComputeEngineParameters(use_gpu=True, gpu_device=count))
