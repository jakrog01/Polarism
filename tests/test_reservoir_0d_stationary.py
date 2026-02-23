import numpy as np

from polarism.reservoir.single_reservoir import SingleReservoir

from tests._helpers import (
    GridCfg,
    Recorder,
    make_grid,
    make_physics_default,
    make_reservoir_cfg,
    pump_uniform,
)


def test_single_reservoir_0d_stationary_with_plot(output_dir):
    grid = make_grid(GridCfg(nx=32, ny=32, lx=200.0, ly=200.0))
    physics = make_physics_default(D=0.0)
    reservoir = SingleReservoir(make_reservoir_cfg(False), physics, grid)

    P0 = 0.02
    psi_amp = 1e-3
    psi = (psi_amp + 0j) * np.ones((grid.nx, grid.ny), dtype=np.complex128)
    P = pump_uniform(P0, grid.X)

    nR_expected = P0 / (physics.gamma_R + physics.R * (psi_amp**2))

    rec = Recorder()
    t = 0.0
    dt = 0.05

    nR_ss = P0 / (physics.gamma_R + physics.R * (psi_amp**2))
    k = physics.gamma_R + physics.R * (psi_amp**2)

    for _ in range(1000):
        reservoir.step(dt, psi, P)
        t += dt
        nR = reservoir.get_reservoir_density()
        nR_mean = float(np.mean(nR))

        nR_expected_t = nR_ss * (1.0 - np.exp(-k * t))
        rel_err = abs(nR_mean - nR_expected_t) / (abs(nR_expected_t) + 1e-30)

        rec.add(t, nR_mean=nR_mean, nR_expected=nR_expected_t, rel_err=rel_err)

    summary = dict(
        nR_expected=nR_expected,
        final_nR_mean=rec.metrics["nR_mean"][-1],
        final_rel_err=rec.metrics["rel_err"][-1],
        max_rel_err=max(rec.metrics["rel_err"]),
    )

    rec.plot(
        output_dir / "reservoir_stationary.png",
        title="SingleReservoir 0D stationary",
        y_keys=["nR_mean", "nR_expected"],
        y_log=False,
    )
    rec.plot(
        output_dir / "reservoir_stationary_error.png",
        title="SingleReservoir 0D stationary relative error",
        y_keys=["rel_err"],
        y_log=True,
    )

    assert summary["final_rel_err"] < 1e-3
