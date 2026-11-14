"""CPU reference checks for two-component condensates."""
from __future__ import annotations

import numpy as np
import pytest

from polarism.config.simulation_parameters import Config
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.results.reader import load_results
from polarism.simulation_controller import SimulationController, _compute_degree_of_polarization, _compute_stokes


def config(*, ndim=1, nx=64, reservoir="none"):
    cfg = Config()
    cfg.grid.ndim = ndim
    cfg.grid.nx = nx
    cfg.grid.ny = nx
    cfg.grid.lx = 8.0
    cfg.grid.ly = 8.0
    cfg.physics.spin = "spinor"
    cfg.physics.hbar = 1.0
    cfg.physics.m_eff = 1.0
    cfg.physics.gamma_C = 0.0
    cfg.physics.g_C = 0.0
    cfg.physics.g_R = 0.0
    cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.noise.enabled = False
    cfg.reservoir.reservoir_type = reservoir
    cfg.laser.P0 = 0.0
    cfg.solver.dt = 1e-3
    cfg.solver.total_time = 1e-3
    return cfg


def test_te_tm_eigenvalues_and_precession():
    cfg = config()
    cfg.physics.zeeman = 0.3
    cfg.physics.te_tm = 0.5
    controller = SimulationController(cfg)
    x = np.asarray(controller.grid.X)
    kx = 2 * np.pi / cfg.grid.lx
    wave = np.exp(1j * kx * x)
    epsilon = kx**2 / 2
    splitting = np.sqrt((cfg.physics.zeeman / 2)**2 + (cfg.physics.te_tm * kx**2)**2)
    discrete_k2 = 4 * np.sin(kx * controller.grid.dx / 2)**2 / controller.grid.dx**2
    off = cfg.physics.te_tm * discrete_k2
    for sign in (1, -1):
        energy = discrete_k2 / 2 + sign * np.sqrt((cfg.physics.zeeman / 2)**2 + off**2)
        vector = np.array([off, energy - discrete_k2 / 2 - cfg.physics.zeeman / 2])
        vector = vector / np.linalg.norm(vector)
        controller.set_initial_state(vector[:, None] * wave)
        controller.solver.advance(controller.model, controller.state, 20)
        overlap = np.vdot(vector[:, None] * wave, controller.state.psi) / wave.size
        measured = -np.angle(overlap) / controller.state.t
        assert measured == pytest.approx(energy, abs=2e-6)
        assert measured == pytest.approx(epsilon + sign * splitting, rel=0.01)
    controller.set_initial_state(np.stack((wave, np.zeros_like(wave))))
    controller.solver.advance(controller.model, controller.state, 100)
    omega = np.sqrt((cfg.physics.zeeman / 2)**2 + off**2)
    expected = (off / omega)**2 * np.sin(omega * controller.state.t)**2
    assert np.mean(np.abs(controller.state.psi_minus)**2) == pytest.approx(expected, abs=2e-6)


def test_te_tm_complex_sign_on_two_dimensional_plane_wave():
    cfg = config(ndim=2, nx=32)
    cfg.physics.te_tm = 0.7
    controller = SimulationController(cfg)
    kx = 2 * np.pi / cfg.grid.lx
    ky = 2 * np.pi / cfg.grid.ly
    wave = np.exp(1j * (kx * controller.grid.X + ky * controller.grid.Y))
    controller.set_initial_state(np.stack((np.zeros_like(wave), wave)))
    rhs, _ = controller.model.rhs(controller.state.field, (), 0)
    cross = np.sin(kx * controller.grid.dx) * np.sin(ky * controller.grid.dy) / (controller.grid.dx * controller.grid.dy)
    assert np.max(np.abs(rhs[0, 0] / wave - (-1j * cfg.physics.te_tm * (-2j * cross)))) < 1e-10


def test_zeeman_splitting_and_decoupled_scalar_runs():
    cfg = config(nx=32)
    cfg.physics.zeeman = 0.4
    controller = SimulationController(cfg)
    initial = np.ones((2, 32), dtype=complex)
    controller.set_initial_state(initial)
    controller.solver.advance(controller.model, controller.state, 50)
    assert controller.state.psi_plus[0] == pytest.approx(np.exp(-0.2j * controller.state.t), abs=1e-12)
    assert controller.state.psi_minus[0] == pytest.approx(np.exp(0.2j * controller.state.t), abs=1e-12)

    cfg.physics.zeeman = 0
    cfg.physics.g_C = 0.2
    cfg.physics.alpha_2 = 0
    spin = SimulationController(cfg)
    x = np.asarray(spin.grid.X)
    initial = np.stack((0.4 + 0.1 * np.cos(x), 0.2 * np.exp(1j * x)))
    spin.set_initial_state(initial)
    spin.solver.advance(spin.model, spin.state, 10)
    scalar_cfg = config(nx=32)
    scalar_cfg.physics.spin = "scalar"
    scalar_cfg.physics.g_C = cfg.physics.g_C
    for component in range(2):
        scalar = SimulationController(scalar_cfg)
        scalar.set_initial_state(initial[component])
        scalar.solver.advance(scalar.model, scalar.state, 10)
        np.testing.assert_allclose(spin.state.psi[component], scalar.state.psi, rtol=0, atol=1e-12)


def test_shared_and_component_reservoirs_and_stokes():
    cfg = config(nx=16, reservoir="single")
    cfg.physics.R = 0.2
    cfg.laser.circular_polarization = 1.0
    controller = SimulationController(cfg)
    field = np.zeros((1, 2, 16), dtype=complex)
    field[:, 0] = 1
    state = (np.ones((1, 2, 16)),)
    per, = controller.model.rhs(field, state, 0)[1]
    assert np.all(per[:, 0] < per[:, 1])
    cfg.reservoir.components = 1
    shared = SimulationController(cfg)
    shared_state = (np.ones((1, 1, 16)),)
    common, = shared.model.rhs(field, shared_state, 0)[1]
    assert common.shape == (1, 1, 16)
    assert np.all(common < per[:, 1:2])

    controller.set_initial_state(np.stack((np.ones(16), np.zeros(16))))
    stokes = _compute_stokes(state=controller.state)
    np.testing.assert_allclose(stokes[0], 1)
    np.testing.assert_allclose(stokes[1:3], 0)
    np.testing.assert_allclose(stokes[3], 1)
    np.testing.assert_allclose(_compute_degree_of_polarization(state=controller.state), 1)
    controller.set_initial_state(np.stack((np.ones(16), np.ones(16))) / np.sqrt(2))
    stokes = _compute_stokes(state=controller.state)
    np.testing.assert_allclose(stokes[0:2], 1)
    np.testing.assert_allclose(stokes[2:4], 0)


def test_incoherent_pump_polarization_and_shared_source():
    cfg = config(nx=16, reservoir="single")
    cfg.laser.laser_type = "uniform"
    cfg.laser.P0 = 0.4
    cfg.laser.circular_polarization = 1.0
    controller = SimulationController(cfg)
    field = np.zeros((1, 2, 16), dtype=complex)
    rates, = controller.model.rhs(field, controller.reservoir.get_state(), 0)[1]
    np.testing.assert_allclose(rates[0, 0], 0.4)
    np.testing.assert_allclose(rates[0, 1], 0)
    cfg.reservoir.components = 1
    shared = SimulationController(cfg)
    rates, = shared.model.rhs(field, shared.reservoir.get_state(), 0)[1]
    np.testing.assert_allclose(rates, 0.4)


def test_spinor_results_and_checkpoint_round_trip(tmp_path):
    cfg = config(nx=16, reservoir="single")
    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.save_psi = True
    cfg.result.output_directory = str(tmp_path)
    cfg.result.observables = ("center-of-mass", "momentum-density")
    controller = SimulationController(cfg)
    controller.run()
    with load_results(controller.run_directory) as result:
        assert result.fields["psi"].shape == (2, 2, 16)
        assert result.fields["density"].shape == (2, 2, 16)
        assert result.fields["stokes"].shape == (2, 4, 16)
        assert result.fields["density_plus"].shape == (2, 16)
        assert result.fields["momentum_density"].shape == (2, 2, 16)
        assert "center_x_plus" in result.scalars
    checkpoint = tmp_path / "spin.h5"
    controller.save_checkpoint(checkpoint)
    restored = SimulationController(cfg)
    restored.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.state.field, controller.state.field)
    np.testing.assert_array_equal(restored.reservoir.get_state()[0], controller.reservoir.get_state()[0])
    cfg.initial_condition.type = "file"
    cfg.initial_condition.path = str(checkpoint)
    from_file = SimulationController(cfg)
    np.testing.assert_array_equal(from_file.state.psi, controller.state.psi)


def test_spinor_initial_and_wigner_noise_are_independent():
    cfg = config(nx=32)
    cfg.noise.enabled = True
    cfg.noise.seed = 123
    cfg.noise.initial = "legacy_positive_uniform"
    controller = SimulationController(cfg)
    assert not np.array_equal(controller.state.psi_plus, controller.state.psi_minus)
    controller.set_initial_state(np.zeros((2, 32), dtype=complex))
    cfg.noise.stochastic = "wigner"
    cfg.physics.gamma_C = 0.1
    noisy = SimulationController(cfg)
    noisy.set_initial_state(np.zeros((2, 32), dtype=complex))
    noisy.model.noise.apply_to_state(noisy.state, noisy.reservoir, 0)
    assert not np.array_equal(noisy.state.psi_plus, noisy.state.psi_minus)
    again = SimulationController(cfg)
    again.set_initial_state(np.zeros((2, 32), dtype=complex))
    again.model.noise.apply_to_state(again.state, again.reservoir, 0)
    np.testing.assert_array_equal(noisy.state.psi, again.state.psi)


def test_batched_spinor_current_keeps_member_and_component_axes():
    cfg = config(nx=16)
    cfg.result.observables = ("current",)
    controller = SimulationController(cfg, members=[{"physics.hbar": 1.0}, {"physics.hbar": 1.2}, {"physics.hbar": 1.4}])
    node = next(node for node in controller._build_result_nodes() if node.name == "current_x")
    current = node.compute_fn(state=controller.state, grid=controller.grid)
    assert current.shape == (3, 2, 16)


def test_drive_jones_vector_normalizes_and_bad_config_rejected():
    cfg = config(nx=16)
    cfg.drive.enabled = True
    cfg.drive.amplitude = 2.0
    cfg.drive.polarization = ["1", "1j"]
    controller = SimulationController(cfg)
    source = controller.model.drive.spatial
    np.testing.assert_allclose(source[0], 2 / np.sqrt(2))
    np.testing.assert_allclose(source[1], 2j / np.sqrt(2))
    cfg.physics.spin = "invalid"
    with pytest.raises(ConfigValidationError, match="physics.spin"):
        validate_config(cfg)
    cfg.physics.spin = "spinor"
    cfg.reservoir.components = 3
    with pytest.raises(ConfigValidationError, match="reservoir.components"):
        validate_config(cfg)
    cfg.reservoir.components = True
    with pytest.raises(ConfigValidationError, match="reservoir.components"):
        validate_config(cfg)
    cfg.reservoir.components = None
    cfg.laser.circular_polarization = 1.1
    with pytest.raises(ConfigValidationError, match="laser.circular_polarization"):
        validate_config(cfg)


@pytest.mark.parametrize("method", ["rk4-fdm-fused", "rk4-cuda"])
def test_unsupported_solvers_reject_spinor(method):
    cfg = config()
    cfg.solver.method = method
    with pytest.raises(ConfigValidationError, match="2 components"):
        validate_config(cfg)


def test_spinor_gpu_backend_rejected_in_cpu_reference_step():
    cfg = config()
    cfg.compute_engine.use_gpu = True
    with pytest.raises(ConfigValidationError, match="use_gpu=False"):
        validate_config(cfg)
