"""CUDA fused-stage and graph checks; run on a CUDA device."""
from pathlib import Path

import numpy as np
import pytest

from polarism.compute_engine import has_cuda_device
from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from polarism.solver.tableaus import TABLEAUS
from tests.unit.conftest import small_config

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not has_cuda_device(), reason="CUDA unavailable")]


def _case(tableau, reservoir_type, precision, eta, method="ifrk-fft-cuda",
          pulse_width=0.03):
    import cupy as cp

    cfg = small_config(
        solver__method=method, solver__tableau=tableau,
        solver__precision=precision, solver__dt=0.007,
        reservoir__reservoir_type=reservoir_type,
        physics__kinetic_relaxation_eta=eta,
        compute_engine=ComputeEngineParameters(use_gpu=True),
    )
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.laser.sigma_time = pulse_width
    cfg.laser.cutoff_sigma = 3.0
    controller = SimulationController(cfg)
    x, y = controller.grid.X, controller.grid.Y
    dtype = cp.complex64 if precision == "single" else cp.complex128
    real = cp.float32 if precision == "single" else cp.float64
    controller.state.psi = cp.asarray(
        0.03 * (1 + 0.1 * cp.cos(2 * cp.pi * x / cfg.grid.lx))
        * cp.exp(1j * 0.2 * cp.sin(2 * cp.pi * y / cfg.grid.ly)), dtype=dtype)
    controller.reservoir.set_state(tuple(
        cp.asarray(0.2 + 0.02 * cp.cos(2 * cp.pi * x / cfg.grid.lx)
                   + index * 0.04, dtype=real)
        for index in range(len(controller.reservoir.get_state()))))
    controller.potential = cp.asarray(
        0.01 * cp.cos(2 * cp.pi * x / cfg.grid.lx)
        - 0.003j * (1 + cp.sin(2 * cp.pi * y / cfg.grid.ly)), dtype=dtype)
    return controller


def _output(controller):
    import cupy as cp

    return (cp.asnumpy(controller.state.psi),
            *(cp.asnumpy(x) for x in controller.reservoir.get_state()))


def _compare(actual, expected, precision):
    tolerance = 1e-5 if precision == "single" else 1e-12
    for a, b in zip(actual, expected):
        relative = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        assert relative < tolerance, relative


@pytest.mark.parametrize("tableau", TABLEAUS)
@pytest.mark.parametrize("reservoir_type", ["single", "double", "quadratic-double"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("eta", [0.0, 0.04])
def test_fused_matches_generic(tableau, reservoir_type, precision, eta):
    fused = _case(tableau, reservoir_type, precision, eta)
    generic = _case(tableau, reservoir_type, precision, eta)
    for step in range(3):
        for controller, use_fused in ((fused, True), (generic, False)):
            controller.state.t = step * controller.cfg.solver.dt
            args = (controller.potential, controller.pump_field, controller.reservoir,
                    controller.boundary_condition, controller.state)
            if use_fused:
                controller.solver.step(*args)
            else:
                controller.solver._generic_step(*args)
    _compare(_output(fused), _output(generic), precision)


@pytest.mark.parametrize("tableau", TABLEAUS)
def test_graph_chunk_matches_repeated_steps(tableau):
    import cupy as cp

    chunk = _case(tableau, "quadratic-double", "double", 0.04, pulse_width=0.3)
    steps = _case(tableau, "quadratic-double", "double", 0.04, pulse_width=0.3)
    args = (chunk.potential, chunk.pump_field, chunk.reservoir,
            chunk.boundary_condition, chunk.state)
    chunk.solver.advance(*args, 0, 131)
    for index in range(131):
        steps.state.t = index * steps.cfg.solver.dt
        steps.solver.step(steps.potential, steps.pump_field, steps.reservoir,
                          steps.boundary_condition, steps.state)
    cp.cuda.Device().synchronize()
    assert chunk.solver._fused._graph_cache
    assert not chunk.solver._fused._graphs_failed
    for actual, expected in zip(_output(chunk), _output(steps)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("reservoir_type", ["single", "quadratic-double"])
@pytest.mark.parametrize("eta", [0.0, 0.04])
def test_alias_matches_old_reference(precision, reservoir_type, eta):
    controller = _case("vern9", reservoir_type, precision, eta,
                       method="ifrk4-fft-cuda")
    assert controller.solver.tableau.name == "rk4"
    for step in range(5):
        controller.state.t = step * controller.cfg.solver.dt
        controller.solver.step(controller.potential, controller.pump_field,
                               controller.reservoir, controller.boundary_condition,
                               controller.state)
    key = f"{precision}_{reservoir_type}_eta{eta:g}"
    path = Path(__file__).parents[1] / "data/ifrk4_reference/reference.npz"
    with np.load(path) as data:
        expected = (data[key + "_psi"],
                    *(data[f"{key}_res{i}"] for i in range(len(controller.reservoir.get_state()))))
        _compare(_output(controller), expected, precision)
