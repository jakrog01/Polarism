"""CPU checks for chunk timing and CUDA configuration."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.compute_engine import has_full_rate_fp64
from polarism.config.simulation_parameters import Config
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.simulation_controller import SimulationController


@pytest.mark.parametrize("capability,expected", [
    ((6, 0), True), ((7, 0), True), ((8, 0), True),
    ((9, 0), True), ((10, 0), True), ((8, 6), False),
    ((8, 9), False), ((7, 5), False), ((6, 1), False),
])
def test_fp64_architecture_classification(capability, expected):
    assert has_full_rate_fp64(*capability) is expected


def test_cuda_graphs_requires_boolean():
    cfg = Config()
    cfg.solver.cuda_graphs = 1
    with pytest.raises(ConfigValidationError, match="cuda_graphs"):
        validate_config(cfg)


def test_controller_chunked_recording_matches_per_step_loop():
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 8
    cfg.solver.method = "rk4-fdm"
    cfg.solver.dt = 0.01
    cfg.solver.total_time = 0.1
    cfg.result.save_interval = 3
    cfg.physics.init_seed = 12
    chunked = SimulationController(cfg)
    baseline = SimulationController(cfg)
    records = []

    class Storage:
        def finalize(self):
            pass

    chunked.storage_visitor = Storage()
    chunked._record_results = lambda t: records.append(
        (t, chunked.state.psi.copy(),
         tuple(field.copy() for field in chunked.reservoir.get_state()))
    )
    chunked.run()

    expected = [(baseline.state.t, baseline.state.psi.copy(),
                 tuple(field.copy() for field in baseline.reservoir.get_state()))]
    baseline.pump_field.lasers = baseline.lasers
    for step in range(10):
        baseline.solver.step(baseline.potential, baseline.pump_field,
                             baseline.reservoir, baseline.boundary_condition,
                             baseline.state)
        baseline.state.t = (step + 1) * cfg.solver.dt
        if (step + 1) % cfg.result.save_interval == 0:
            expected.append((baseline.state.t, baseline.state.psi.copy(),
                             tuple(field.copy() for field in baseline.reservoir.get_state())))
    assert len(records) == len(expected)
    for actual, old in zip(records, expected):
        assert actual[0] == old[0]
        np.testing.assert_array_equal(actual[1], old[1])
        for field, other in zip(actual[2], old[2]):
            np.testing.assert_array_equal(field, other)
