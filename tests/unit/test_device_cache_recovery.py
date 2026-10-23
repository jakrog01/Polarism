"""Cache markers must only describe complete device state."""
import gc
import weakref

import numpy as np
import pytest

from polarism.compute_engine import compute_engine, has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters, Config
from polarism.solver.cuda_pump import CudaPumpCombiner
from polarism.solver.ifrk_cuda_fused import FusedIFRK
from polarism.solver.rk4_cuda_solver import RK4CudaSolver
from tests.unit.test_buffer_reallocation import _case, _run, _assert_matches_fresh


class _Kernel:
    def compile(self):
        pass


class _ArrayModule:
    float32 = np.float32
    ascontiguousarray = staticmethod(np.ascontiguousarray)
    concatenate = staticmethod(np.concatenate)
    stack = staticmethod(np.stack)
    empty = staticmethod(np.empty)
    RawKernel = staticmethod(lambda *args, **kwargs: _Kernel())

    def __init__(self):
        self.fail_offsets = False

    def asarray(self, value, dtype=None):
        if self.fail_offsets and np.asarray(value).dtype == np.int32:
            self.fail_offsets = False
            raise MemoryError("injected offsets allocation failure")
        return np.asarray(value, dtype=dtype)


def test_pump_combiner_rebuilds_after_partial_failure():
    xp = _ArrayModule()
    combiner = CudaPumpCombiner(xp, np.float32, 8)
    first = np.ones((2, 2), np.float32)
    second = np.full((2, 2), 2, np.float32)
    combiner.prepare(((first, lambda t: 1),), 1, 1, shape=(2, 1, 2, 2))
    xp.fail_offsets = True
    with pytest.raises(MemoryError, match="injected offsets allocation failure"):
        combiner.prepare(((second, lambda t: 1),), 1, 1, shape=(2, 1, 2, 2))
    combiner.prepare(((first, lambda t: 1),), 1, 1, shape=(2, 1, 2, 2))
    np.testing.assert_array_equal(combiner.env, first.ravel())


def test_ifrk_potential_cache_keeps_its_identity_alive():
    fused = FusedIFRK.__new__(FusedIFRK)
    fused.xp = np
    fused.real = np.float64
    fused._potential_cached = None
    fused._potential_id = None
    fused._graph_cache = {}
    fused._zero_pump = np.zeros((2, 2))
    potential = np.ones((2, 2), np.complex128)
    reference = weakref.ref(potential)
    fused._potential(potential)
    del potential
    gc.collect()
    assert reference() is not None


def test_ifrk_zero_pump_follows_batch_shape():
    fused = FusedIFRK.__new__(FusedIFRK)
    fused.xp = np
    fused.real = np.float64
    fused.shape = (1, 1, 2, 2)
    fused._batched = False
    fused.n = 4
    fused._zero_pump = np.zeros((1, 1, 2, 2))
    fused._graph_cache = {}
    fused._graph_signature = None
    fused._pump_combiner = None
    fused._potential_cached = None
    state = type("State", (), {"field": np.zeros((3, 1, 2, 2)), "field_k": None,
                               "batched": True})()
    fused._activate_shape(state)
    assert fused._zero_pump.shape == state.field.shape


def test_rk4_advance_buffer_failure_does_not_mark_cache_valid():
    class _StreamModule:
        def __init__(self):
            self.fail = True

        def Stream(self, non_blocking):
            if self.fail:
                self.fail = False
                raise MemoryError("injected stream allocation failure")
            return object()

    streams = _StreamModule()
    xp = type("XP", (), {"empty": staticmethod(np.empty),
                          "cuda": type("Cuda", (), {"Stream": streams.Stream})()})()
    solver = RK4CudaSolver.__new__(RK4CudaSolver)
    solver.xp = xp
    solver.N = 4
    solver._members = 2
    solver._reservoir_type = "single"
    solver._complex_dtype = np.complex128
    solver._real_dtype = np.float64
    with pytest.raises(MemoryError, match="injected stream allocation failure"):
        solver._ensure_advance_buffers()
    assert not hasattr(solver, "_advance_buffers")
    solver._ensure_advance_buffers()
    assert solver._advance_buffers[0][0].size == 8


def test_rk4_batch_retries_failed_stage_allocation():
    solver = RK4CudaSolver.__new__(RK4CudaSolver)
    solver.config = Config()
    solver.batch_physics = type("Physics", (), {"host": {}})()
    solver._physics_fields = ()
    solver._batched = False
    solver._members = 1
    solver._batch_ready = True
    solver._graph_cache = {}
    solver._graph_signature = None
    solver._tiled_kernels = {}
    calls = 0

    def allocate():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MemoryError("injected stage allocation failure")

    solver._compile_kernels = lambda: None
    solver._init_gpu_buffers = allocate
    with pytest.raises(MemoryError, match="injected stage allocation failure"):
        solver._activate_batch((2, 1, 2, 2), True)
    solver._activate_batch((2, 1, 2, 2), True)
    assert calls == 2


@pytest.mark.gpu
def test_rk4_batch_allocation_failure_retries():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    solver, args = _case("rk4-cuda", 4, cp)
    _run(solver, args, "rk4-cuda")
    _, small_args = _case("rk4-cuda", 2, cp)
    original = cp.empty
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise MemoryError("injected batch allocation failure")
        return original(*args, **kwargs)

    try:
        cp.empty = fail_once
        with pytest.raises(MemoryError, match="injected batch allocation failure"):
            solver._activate_batch(small_args[-1].field.shape, small_args[-1].batched)
    finally:
        cp.empty = original
    assert not solver._batch_ready
    _assert_matches_fresh(solver, small_args, "rk4-cuda", 2, cp)


@pytest.mark.gpu
def test_rk4_batch_shape_changes_match_fresh():
    if not has_cuda_device():
        pytest.skip("CUDA device unavailable")
    import cupy as cp

    compute_engine.configure(ComputeEngineParameters(use_gpu=True))
    solver, args = _case("rk4-cuda", 4, cp)
    _assert_matches_fresh(solver, args, "rk4-cuda", 4, cp)
    for batch in (2, 4):
        _, args = _case("rk4-cuda", batch, cp)
        _assert_matches_fresh(solver, args, "rk4-cuda", batch, cp)
