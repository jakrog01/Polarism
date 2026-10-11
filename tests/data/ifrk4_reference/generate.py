"""Regenerate reference trajectories with the pre-engine IFRK4 implementation.

Run from the repository root with ``PYTHONPATH=. .venv/bin/python
tests/data/ifrk4_reference/generate.py`` at commit
``3f83ad37675c60c7d666846969a46f81905f52b7`` before replacing the solver.
The checked-in ``reference.npz`` was generated on CPU with NumPy.
"""
from pathlib import Path

import numpy as np

from polarism.config.simulation_parameters import ComputeEngineParameters
from polarism.simulation_controller import SimulationController
from tests.unit.conftest import small_config


def run_cases():
    output = {}
    for precision in ("single", "double"):
        for reservoir_type in ("single", "quadratic-double"):
            for eta in (0.0, 0.04):
                cfg = small_config(
                    solver__method="ifrk4-fft-cuda",
                    solver__precision=precision,
                    solver__dt=0.007,
                    solver__total_time=0.035,
                    reservoir__reservoir_type=reservoir_type,
                    physics__kinetic_relaxation_eta=eta,
                    compute_engine=ComputeEngineParameters(use_gpu=False),
                )
                cfg.laser.laser_type = "pulse-gaussian"
                cfg.laser.sigma_time = 0.03
                cfg.laser.cutoff_sigma = 3.0
                controller = SimulationController(cfg)
                x, y = controller.grid.X, controller.grid.Y
                dtype = np.complex64 if precision == "single" else np.complex128
                real = np.float32 if precision == "single" else np.float64
                controller.state.psi = np.asarray(
                    0.03 * (1 + 0.1 * np.cos(2 * np.pi * x / cfg.grid.lx))
                    * np.exp(1j * 0.2 * np.sin(2 * np.pi * y / cfg.grid.ly)), dtype=dtype
                )
                fields = tuple(np.asarray(0.2 + 0.02 * np.cos(2 * np.pi * x / cfg.grid.lx)
                                          + index * 0.04, dtype=real)
                               for index in range(len(controller.reservoir.get_state())))
                controller.reservoir.set_state(fields)
                controller.potential = np.asarray(
                    0.01 * np.cos(2 * np.pi * x / cfg.grid.lx)
                    - 0.003j * (1 + np.sin(2 * np.pi * y / cfg.grid.ly)), dtype=dtype
                )
                for step in range(5):
                    controller.state.t = step * cfg.solver.dt
                    controller.solver.step(controller.potential, controller.pump_field,
                                           controller.reservoir, controller.boundary_condition,
                                           controller.state)
                key = f"{precision}_{reservoir_type}_eta{eta:g}"
                output[key + "_psi"] = np.asarray(controller.state.psi)
                for index, field in enumerate(controller.reservoir.get_state()):
                    output[f"{key}_res{index}"] = np.asarray(field)
    return output


def generate():
    np.savez_compressed(Path(__file__).with_name("reference.npz"), **run_cases())


if __name__ == "__main__":
    generate()
