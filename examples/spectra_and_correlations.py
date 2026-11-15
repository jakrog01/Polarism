"""Emission spectrum E(k), momentum distribution and g1 of a pumped condensate.

A Gaussian spot above threshold is integrated with adaptive DOP853 steps.
After the condensate has formed, accumulators sample the field every 0.1 ps
and the script plots the spectrum against the bare dispersion, marks the
emission energy and shows the radial momentum distribution and |g1|.
"""
import matplotlib.pyplot as plt
import numpy as np

import polarism as ps
from polarism.compute_engine import compute_engine


def main() -> None:
    cfg = ps.Config()
    cfg.grid.nx = cfg.grid.ny = 128
    cfg.grid.lx = cfg.grid.ly = 80.0
    cfg.grid.grid_type = "periodic"

    cfg.physics.g_C = 0.01
    cfg.physics.g_R = 0.02
    cfg.reservoir.reservoir_type = "double"
    cfg.laser.mode = "single"
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.P0 = 0.3
    cfg.laser.sigma_space = 8.0

    cfg.solver.method = "ifrk-fft-cuda"
    cfg.solver.tableau = "dop853"
    cfg.solver.adaptive = True
    cfg.solver.dt = 0.1
    cfg.solver.rtol = 1e-6
    cfg.solver.total_time = 600.0

    cfg.result.save_results = False
    cfg.result.accumulators = {
        "spectrum-k": {"start": 200.0, "every": 0.1, "segment": 2048},
        "momentum-distribution": {"start": 200.0, "every": 1.0},
    }

    cfg.compute_engine.use_gpu = False
    compute_engine.configure(cfg.compute_engine)

    controller = ps.SimulationController(cfg)
    controller.run()
    accumulated = controller.accumulated
    print("adaptive steps:", controller.step_statistics)

    spectrum = accumulated["spectrum_k"].value
    energy = accumulated["spectrum_k_energy"].value
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(cfg.grid.nx, cfg.grid.lx / cfg.grid.nx))
    mu = energy[np.argmax(spectrum.sum(axis=-1))]
    print(f"emission energy mu = {mu:.4f} meV")

    n_k = accumulated["momentum_distribution"].value
    g1 = np.abs(accumulated["g1"].value)
    x = np.fft.fftshift(np.fft.fftfreq(cfg.grid.nx, 1 / cfg.grid.lx))

    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 4.2))
    left.pcolormesh(kx, energy, np.log10(spectrum + spectrum.max() * 1e-6), shading="auto")
    bare = cfg.physics.hbar ** 2 * kx ** 2 / (2 * cfg.physics.m_eff)
    left.plot(kx, bare, "w--", lw=1, label="bare dispersion")
    left.axhline(mu, color="tab:red", lw=1, label=f"μ = {mu:.3f} meV")
    left.set(xlim=(-3, 3), ylim=(mu - 1.0, mu + 2.0), xlabel="kx (µm⁻¹)", ylabel="E (meV)",
             title="log₁₀ S(E, kx, ky=0)")
    left.legend(loc="upper right", fontsize=8)
    middle.semilogy(kx, n_k[cfg.grid.ny // 2])
    middle.set(xlim=(-3, 3), xlabel="kx (µm⁻¹)", ylabel="n(k) (particles/mode)",
               title="Momentum distribution, ky = 0")
    right.plot(x, g1[cfg.grid.ny // 2])
    right.set(xlabel="Δx (µm)", ylabel="|g1(Δx)|", ylim=(0, 1.05), title="First-order coherence")
    figure.tight_layout()
    figure.savefig("spectra_and_correlations.png", dpi=150)


if __name__ == "__main__":
    main()
