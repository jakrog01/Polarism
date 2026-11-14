"""Extra term and dispersion contracts with test-only plugins."""
from __future__ import annotations

import numpy as np
import pytest

from polarism import Config, SimulationController, Term
from polarism.config.validation import ConfigValidationError, validate_config
from polarism.dispersion import (
    Dispersion,
    ParabolicDispersion,
    available_dispersions,
    register_dispersion,
)
from polarism.model_terms import CudaSnippet, available_terms, register_term
from polarism.solver.solver_compatibility import SolverCompatibilityError, check_solver_compatibility
from polarism.sweep import Sweep
from polarism.sweep_config import apply_overrides, override_kind

TERM_SOLVERS = ("rk4-fdm", "rk4-cuda", "ip-rk4", "etd-rk2", "split-step-fft")
SPECTRAL_SOLVERS = ("split-step-fft", "ip-rk4", "etd-rk2")
NO_TERM_SOLVERS = ("rk4-fdm-fused",)
PARABOLIC_ONLY_SOLVERS = ("rk4-fdm", "split-step-fft-cuda", "ifrk-fft-cuda", "ifrk4-fft-cuda") + NO_TERM_SOLVERS
LOSS_RTOL = {"rk4-fdm": 1e-8, "rk4-cuda": 1e-8, "ip-rk4": 1e-8, "etd-rk2": 1e-4, "split-step-fft": 5e-3}
SHIFT_ATOL = {"rk4-fdm": 1e-8, "rk4-cuda": 1e-8, "ip-rk4": 1e-8, "etd-rk2": 1e-6, "split-step-fft": 1e-8}


class NonlinearLoss(Term):
    category = "rate"
    cuda = CudaSnippet("-0.5 * alpha * rho", (("double", "alpha"),))

    def coefficient(self, field, reservoir, reservoir_fields, t):
        return -0.5 * self.params["alpha"] * abs(field) ** 2


class ConstantShift(Term):
    category = "energy"
    cuda = CudaSnippet("energy", (("double", "energy"),))

    def coefficient(self, field, reservoir, reservoir_fields, t):
        return self.params["energy"]


class Quartic(Dispersion):
    def energy(self, grid, physics):
        hbar, m_eff = physics.get("hbar"), physics.get("m_eff")
        return hbar**2 * grid.k_squared / (2 * m_eff) + 0.5 * grid.k_squared**2


@pytest.fixture
def plugins():
    terms, dispersions = dict(available_terms), dict(available_dispersions)
    register_term("nonlinear-loss")(NonlinearLoss)
    register_term("constant-shift")(ConstantShift)
    register_dispersion("quartic")(Quartic)
    yield
    for registry, saved in ((available_terms, terms), (available_dispersions, dispersions)):
        registry.clear()
        registry.update(saved)


def config(method="rk4-fdm", reservoir="none"):
    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = 16
    cfg.grid.lx = cfg.grid.ly = 16.0
    cfg.solver.method = method
    cfg.solver.dt = 0.001
    cfg.solver.total_time = 0.5
    cfg.solver.cuda_graphs = False
    cfg.reservoir.reservoir_type = reservoir
    cfg.laser.P0 = 0.0
    cfg.physics.g_C = cfg.physics.g_R = cfg.physics.g_I = 0.0
    cfg.physics.R = 0.0
    cfg.physics.gamma_C = 0.0
    cfg.noise.seed = 12
    cfg.noise.initial = "none"
    cfg.result.save_results = False
    cfg.compute_engine.use_gpu = False
    return cfg


def uniform(run, amplitude=1.0):
    run.state.psi = np.full(run.grid.shape, amplitude, dtype=np.complex128)


@pytest.mark.parametrize("method", TERM_SOLVERS)
def test_nonlinear_loss_matches_analytic_decay(plugins, method):
    cfg = config(method)
    cfg.solver.total_time = 1.0
    cfg.terms = {"nonlinear-loss": {"alpha": 1.0}}
    run = SimulationController(cfg)
    assert [type(term) for term in run.model.terms] == [NonlinearLoss]
    uniform(run)
    run.run()
    expected = 1.0 / (1.0 + 1.0 * 1.0 * cfg.solver.total_time)
    np.testing.assert_allclose(np.abs(run.state.psi) ** 2, expected, rtol=LOSS_RTOL[method], atol=0)


@pytest.mark.parametrize("method", TERM_SOLVERS)
def test_energy_term_rotates_phase(plugins, method):
    cfg = config(method)
    cfg.terms = {"constant-shift": {"energy": 1.0}}
    run = SimulationController(cfg)
    uniform(run)
    run.run()
    expected = np.exp(-1j * 1.0 * cfg.solver.total_time / cfg.physics.hbar)
    np.testing.assert_allclose(run.state.psi, expected, rtol=0, atol=SHIFT_ATOL[method])


def test_terms_sum_in_configuration_order(plugins):
    cfg = config()
    cfg.terms = {"constant-shift": {"energy": 2.0}, "nonlinear-loss": {"alpha": 1.0}}
    run = SimulationController(cfg)
    field = np.full((1, 1, *run.grid.shape), 2.0, dtype=np.complex128)
    energy, rate = run.model.extra_coefficients(field, (), 0.0)
    assert energy == 2.0
    np.testing.assert_allclose(rate, -2.0)


def test_term_category_is_checked():
    class Misnamed(Term):
        category = "power"

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return 0.0

    with pytest.raises(ValueError, match="category"):
        Misnamed({}, None, None)


def test_rk4_cuda_rejects_term_without_cuda_snippet(plugins):
    class CpuOnly(Term):
        category = "rate"

        def coefficient(self, field, reservoir, reservoir_fields, t):
            return 0.0

    register_term("cpu-only")(CpuOnly)
    cfg = config("rk4-cuda")
    cfg.terms = {"cpu-only": {}}
    with pytest.raises(SolverCompatibilityError, match="CUDA snippet"):
        check_solver_compatibility(cfg)


@pytest.mark.parametrize("method", SPECTRAL_SOLVERS)
def test_plane_wave_follows_dispersion_phase(plugins, method):
    cfg = config(method)
    cfg.solver.dt = 0.01
    cfg.solver.total_time = 1.0
    cfg.physics.dispersion = "quartic"
    run = SimulationController(cfg)
    assert run.model.dispersion.kind == "quartic"
    psi_k = np.zeros(run.grid.shape, dtype=np.complex128)
    psi_k[2, 3] = 1.0
    run.state.psi = np.fft.ifft2(psi_k)
    run.run()
    hbar, m_eff = cfg.physics.hbar, cfg.physics.m_eff
    k2 = run.grid.k_squared
    parabolic = hbar**2 * k2 / (2 * m_eff)
    energy = parabolic + 0.5 * k2**2
    expected = np.fft.ifft2(psi_k * np.exp(-1j * energy * cfg.solver.total_time / hbar))
    wrong = np.fft.ifft2(psi_k * np.exp(-1j * parabolic * cfg.solver.total_time / hbar))
    np.testing.assert_allclose(run.state.psi, expected, rtol=0, atol=1e-12)
    assert not np.allclose(run.state.psi, wrong, rtol=0, atol=1e-6)


def test_parabolic_dispersion_is_default():
    run = SimulationController(config())
    assert isinstance(run.model.dispersion, ParabolicDispersion)
    assert run.model.dispersion.kind == "parabolic"
    assert run.model.terms == ()
    hbar, m_eff = run.cfg.physics.hbar, run.cfg.physics.m_eff
    np.testing.assert_allclose(
        run.model.dispersion.energy(run.grid, run.model.physics),
        hbar**2 * run.grid.k_squared / (2 * m_eff),
    )


def test_unknown_term_and_dispersion_rejected_at_validation(plugins):
    cfg = config()
    cfg.terms = {"missing-term": {}}
    with pytest.raises(ConfigValidationError, match="missing-term") as exc:
        validate_config(cfg)
    assert "nonlinear-loss" in str(exc.value)
    cfg.terms = {"nonlinear-loss": 1.0}
    with pytest.raises(ConfigValidationError, match="mapping"):
        validate_config(cfg)
    cfg.terms = {}
    cfg.physics.dispersion = "missing-dispersion"
    with pytest.raises(ConfigValidationError, match="physics.dispersion") as exc:
        validate_config(cfg)
    assert "quartic" in str(exc.value)


@pytest.mark.parametrize("method", NO_TERM_SOLVERS)
def test_solvers_without_terms_reject_them(plugins, method):
    cfg = config(method, "single")
    cfg.terms = {"nonlinear-loss": {"alpha": 1.0}}
    with pytest.raises(SolverCompatibilityError, match="extra terms"):
        check_solver_compatibility(cfg)


@pytest.mark.parametrize("method", TERM_SOLVERS)
def test_term_solvers_accept_terms(plugins, method):
    cfg = config(method, "single")
    cfg.terms = {"nonlinear-loss": {"alpha": 1.0}}
    check_solver_compatibility(cfg)


@pytest.mark.parametrize("method", PARABOLIC_ONLY_SOLVERS)
def test_parabolic_only_solvers_reject_other_dispersion(plugins, method):
    cfg = config(method, "single")
    cfg.physics.dispersion = "quartic"
    with pytest.raises(SolverCompatibilityError, match="dispersion"):
        check_solver_compatibility(cfg)


@pytest.mark.parametrize("method", SPECTRAL_SOLVERS)
def test_spectral_solvers_accept_other_dispersion(plugins, method):
    cfg = config(method, "single")
    cfg.physics.dispersion = "quartic"
    check_solver_compatibility(cfg)


def test_sweep_path_kinds():
    assert override_kind("physics.dispersion") == "group"
    assert override_kind("physics.g_C") == "batch"
    assert override_kind("terms.nonlinear-loss.alpha") == "group"


def test_sweep_overrides_term_parameters_in_a_copy(plugins):
    cfg = config()
    cfg.terms = {"nonlinear-loss": {"alpha": 1.0}}
    changed = apply_overrides(cfg, {"terms.nonlinear-loss.alpha": 3.0}, allow_groups=True)
    assert changed.terms == {"nonlinear-loss": {"alpha": 3.0}}
    assert cfg.terms == {"nonlinear-loss": {"alpha": 1.0}}
    with pytest.raises(ValueError, match="non-batchable"):
        apply_overrides(cfg, {"terms.nonlinear-loss.alpha": 3.0})
    with pytest.raises(ValueError, match="Unknown sweep path"):
        apply_overrides(cfg, {"terms.nonlinear-loss.beta": 3.0}, allow_groups=True)
    with pytest.raises(ValueError, match="Unknown sweep path"):
        apply_overrides(cfg, {"terms.constant-shift.energy": 3.0}, allow_groups=True)


def test_sweeping_term_parameters_and_dispersion_forms_groups(plugins):
    cfg = config("split-step-fft")
    cfg.terms = {"nonlinear-loss": {"alpha": 1.0}}
    sweep = Sweep(cfg, {"terms.nonlinear-loss.alpha": [0.5, 1.0],
                        "physics.dispersion": ["parabolic", "quartic"]})
    assert [group["indices"] for group in sweep.groups] == [[0], [1], [2], [3]]


def test_terms_and_dispersion_round_trip_through_yaml(plugins, tmp_path):
    cfg = config("split-step-fft")
    cfg.terms = {"nonlinear-loss": {"alpha": 0.25}}
    cfg.physics.dispersion = "quartic"
    restored = Config.from_yaml(cfg.to_yaml(tmp_path / "run.yaml"))
    assert restored.terms == {"nonlinear-loss": {"alpha": 0.25}}
    assert restored.physics.dispersion == "quartic"
