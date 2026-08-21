from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.signal import find_peaks

from mnist_digits_polariton_snn_dynamic.config.loader import (
    load_polarism_config,
    load_snn_dynamic_config,
)
from mnist_digits_polariton_snn_dynamic.simulation.calibration import final_power_max
from mnist_digits_polariton_snn_dynamic.simulation.spike_threshold import (
    NoSpikingRegimeError,
    SpikeThresholdSettings,
    count_upward_crossings,
    find_spike_threshold,
    integrate_zero_dim,
    select_threshold_power,
)
from polarism.config.simulation_parameters import GridParameters, PhysicsConstants, ReservoirParameters
from polarism.compute_engine import has_cuda_device
from polarism.grid.create_grid import create_grid
from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
from mnist_digits_polariton_snn_dynamic.config.loader import build_geometry
from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources
from mnist_digits_polariton_snn_dynamic.simulation.runner import simulate_one_image


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_nr_crit_matches_solver_gain_loss() -> None:
    physics = load_polarism_config(str(PACKAGE_DIR / "polarism_base.yaml")).physics
    critical = physics.gamma_C / physics.R
    values = np.array((critical * 0.9, critical, critical * 1.1))
    assert physics.R * values[0] < physics.gamma_C < physics.R * values[2]
    assert np.isclose(critical, physics.gamma_C / physics.R)


def test_count_upward_crossings_synthetic() -> None:
    time = np.linspace(0.0, 6.0 * np.pi, 6001)
    result = count_upward_crossings(time, np.sin(time - np.pi / 2.0 + 0.17), hysteresis=0.02)
    assert result.n_crossings == 3
    assert any(abs(value / (time[1] - time[0]) - round(value / (time[1] - time[0]))) > 1.0e-6 for value in result.crossing_times_ps[1:])
    chatter = count_upward_crossings(time, 0.01 * np.sin(time), hysteresis=0.02)
    assert chatter.n_crossings == 0
    assert count_upward_crossings(time, np.ones_like(time), hysteresis=0.02).n_crossings == 1
    assert count_upward_crossings(time, -np.ones_like(time), hysteresis=0.02).n_crossings == 0


def test_zero_dim_matches_quadratic_double_reservoir() -> None:
    physics = PhysicsConstants(R=0.023, gamma_R=0.15, gamma_I=0.001, kappa=0.05)
    grid = create_grid(GridParameters(nx=16, ny=16, lx=8.0, ly=8.0))
    reservoir = QuadraticDoubleReservoir(ReservoirParameters(reservoir_type="quadratic-double"), physics, grid)
    time = np.arange(0.0, 2.01, 0.01)
    pump = 0.7 + 0.2 * np.sin(time)
    _, expected = integrate_zero_dim(time, pump, physics)
    psi = np.zeros((grid.ny, grid.nx), dtype=np.complex128)
    state = reservoir.get_state()
    observed = [float(state[0][0, 0])]
    for index, step in enumerate(np.diff(time)):
        p0 = np.full((grid.ny, grid.nx), pump[index])
        p1 = np.full((grid.ny, grid.nx), pump[index + 1])
        pm = 0.5 * (p0 + p1)
        k1 = reservoir.get_derivatives(psi, p0, state)
        k2 = reservoir.get_derivatives(psi, pm, tuple(value + 0.5 * step * delta for value, delta in zip(state, k1)))
        k3 = reservoir.get_derivatives(psi, pm, tuple(value + 0.5 * step * delta for value, delta in zip(state, k2)))
        k4 = reservoir.get_derivatives(psi, p1, tuple(value + step * delta for value, delta in zip(state, k3)))
        state = tuple(value + step * (a + 2.0 * b + 2.0 * c + d) / 6.0 for value, a, b, c, d in zip(state, k1, k2, k3, k4))
        observed.append(float(state[0][0, 0]))
    np.testing.assert_allclose(np.asarray(observed), expected, rtol=1.0e-6, atol=1.0e-10)


def test_plateau_selection_is_grid_independent() -> None:
    def evaluate(power: float) -> int:
        return 4 if 10.0 <= power <= 40.0 else 3
    results = []
    for number in (48, 192):
        powers = np.geomspace(1.0, 100.0, number)
        counts = np.array([evaluate(value) for value in powers])
        results.append(select_threshold_power(powers, counts, evaluate, edge_tol_rel=1.0e-4))
    for p_lo, p_hi, threshold, maximum in results:
        assert maximum == 4
        assert abs(p_lo - 10.0) < 0.01
        assert abs(p_hi - 40.0) < 0.04
        assert abs(threshold - 20.0) < 0.02


def test_threshold_on_mnist_base_config(tmp_path: Path) -> None:
    result = find_spike_threshold(str(PACKAGE_DIR / "config.yaml"), "base", str(tmp_path), SpikeThresholdSettings())
    assert result.n_crossings_max == 9
    assert 170.0 < result.plateau["P_lo"] < 171.0
    assert 192.0 < result.plateau["P_hi"] < 194.0
    assert 178.0 < result.P_threshold < 185.0
    assert abs(result.nR_crit - 4.3478) < 1.0e-3
    assert (tmp_path / "spike_threshold.png").exists()
    assert json.loads((tmp_path / "spike_threshold.json").read_text())["final_power_max"] == result.P_threshold
    assert final_power_max(str(tmp_path / "spike_threshold.json")) == result.P_threshold


def test_no_spiking_regime_is_hard_error(tmp_path: Path) -> None:
    with pytest.raises(NoSpikingRegimeError) as caught:
        find_spike_threshold(str(PACKAGE_DIR / "config.yaml"), "low", str(tmp_path), SpikeThresholdSettings(p_max=1.0))
    assert caught.value.result.status == "no_spiking_regime"
    assert json.loads((tmp_path / "spike_threshold.json").read_text())["status"] == "no_spiking_regime"


def test_diffusion_config_is_rejected(tmp_path: Path) -> None:
    dynamic = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
    polarism = yaml.safe_load((PACKAGE_DIR / "polarism_base.yaml").read_text())
    polarism["physics"]["reservoir_diffusion_R"] = 1.0e-3
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(polarism), encoding="utf-8")
    dynamic["polarism_config_path"] = "base.yaml"
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(dynamic), encoding="utf-8")
    with pytest.raises(ValueError, match="reservoir_diffusion"):
        find_spike_threshold(str(tmp_path / "scenario.yaml"), "diffusive", str(tmp_path / "output"))


def test_threshold_section_is_optional() -> None:
    for path in (PACKAGE_DIR / "scenarios" / "pitch_sigma_sweep").glob("*.yaml"):
        if path.name != "manifest.yaml":
            assert load_snn_dynamic_config(str(path)).threshold.p_min == 1.0


@pytest.mark.gpu
@pytest.mark.slow
def test_predicted_threshold_produces_spiking_in_full_gpe(tmp_path: Path) -> None:
    if not has_cuda_device():
        pytest.skip("CUDA device is required")
    dynamic = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
    polarism = yaml.safe_load((PACKAGE_DIR / "polarism_base.yaml").read_text())
    dynamic["polarism_config_path"] = "base.yaml"
    dynamic["geometry"].update({"n_side": 1, "pitch_um": 20.0})
    dynamic["pulse"].update({"n_pulses": 5})
    dynamic["readout"].update({"stride_steps": 1, "batch_size": 1})
    dynamic["threshold"] = {"p_min": 20.0, "p_max": 500.0, "n_points": 64}
    polarism["grid"].update({"nx": 64, "ny": 64, "lx": 40.0, "ly": 40.0})
    polarism["solver"].update({"total_time": 60.0, "dt": 0.02})
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(polarism), encoding="utf-8")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(dynamic), encoding="utf-8")
    result = find_spike_threshold(str(scenario_path), "gpu", str(tmp_path / "threshold"), SpikeThresholdSettings(p_min=20.0, p_max=500.0, n_points=64))
    loaded = load_snn_dynamic_config(str(scenario_path))
    resources = SharedSimResources(load_polarism_config(loaded.polarism_config_path), build_geometry(loaded.geometry), loaded.pulse, loaded.readout.mask_radius_um)
    powers = (result.plateau["P_lo"] / 2.0, result.P_threshold, 2.0 * result.plateau["P_hi"])
    traces = [simulate_one_image(resources, np.array((power,)), 0.0, 1, False).psi[:, 0] for power in powers]
    density_floor = float(resources.cfg.physics.init_eps) ** 2
    condensed = [trace > density_floor * 1.0e3 for trace in traces]
    counts = [len(find_peaks(trace, height=density_floor * 1.0e3, prominence=density_floor * 1.0e2)[0]) for trace in traces]
    assert not np.any(condensed[0])
    assert np.any(condensed[1])
    assert counts[1] > counts[0]
    assert counts[1] > counts[2]
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for power, trace in zip(powers, traces):
        axis.plot(trace, label=f"P={power:.3g}")
    axis.legend()
    figure.savefig(tmp_path / "full_gpe_threshold_comparison.png", dpi=150)
    plt.close(figure)
