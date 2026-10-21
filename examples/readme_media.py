"""Render the README animation: a ring-shaped pump trapping vortices.

Runs on a CUDA GPU and writes docs/assets/readme/ring_vortices.{webp,png}.
"""
from pathlib import Path

import cupy as cp
import numpy as np
from matplotlib import colormaps
from PIL import Image

import polarism as ps
from polarism.compute_engine import compute_engine
from polarism.laser import register_laser
from polarism.laser.abstract_laser import AbstractLaser

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "readme"
FRAME_EVERY = 100
FIRST_FRAME = 60
N_FRAMES = 180
CROP = 0.62
PANEL = 300
GAP = 8
BACKGROUND = (13, 17, 23)


@register_laser("ring")
class RingLaser(AbstractLaser):
    """Continuous Gaussian ring of radius `ring_radius` and width `sigma_space`."""

    def __init__(self, laser_config, X, Y, precision="double"):
        super().__init__(laser_config, X, Y, precision)
        self.radius = laser_config.extra.get("ring_radius", 20.0)
        self.width = laser_config.sigma_space
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        r = self.xp.sqrt((X - self.x0) ** 2 + (Y - self.y0) ** 2)
        return self.xp.exp(-0.5 * ((r - self.radius) / self.width) ** 2)

    def _P_time(self, t):
        return 1.0


def make_config():
    cfg = ps.Config()
    cfg.grid.nx = cfg.grid.ny = 256
    cfg.grid.lx = cfg.grid.ly = 100.0
    cfg.grid.grid_type = "periodic"
    cfg.reservoir.reservoir_type = "single"
    cfg.laser.mode = "single"
    cfg.laser.laser_type = "ring"
    cfg.laser.extra = {"ring_radius": 20.0}
    cfg.laser.P0 = 0.1
    cfg.laser.sigma_space = 4.0
    cfg.physics.init_mode = "complex_gaussian_zero_mean"
    cfg.physics.init_seed = 5
    cfg.solver.method = "ifrk-fft-cuda"
    cfg.solver.precision = "single"
    cfg.solver.dt = 0.05
    cfg.compute_engine.use_gpu = True
    return cfg


def colorize(values, cmap, lo, hi):
    rgba = colormaps[cmap]((values - lo) / (hi - lo), bytes=True)
    return Image.fromarray(rgba[..., :3]).resize((PANEL, PANEL), Image.BICUBIC)


def render(psi):
    n = psi.shape[0]
    a = int(n * (1 - CROP) / 2)
    psi = psi[a:n - a, a:n - a]
    psi = psi * np.exp(-1j * np.angle(psi.sum()))
    density = np.abs(psi) ** 2
    frame = Image.new("RGB", (2 * PANEL + GAP, PANEL), BACKGROUND)
    frame.paste(colorize(density, "inferno", 0.0, density.max()), (0, 0))
    frame.paste(colorize(np.angle(psi), "twilight", -np.pi, np.pi), (PANEL + GAP, 0))
    return frame


def main():
    cfg = make_config()
    compute_engine.configure(cfg.compute_engine)
    sim = ps.SimulationController(cfg)
    sim.pump_field.lasers = sim.lasers

    frames = []
    step = 0
    for i in range(FIRST_FRAME + N_FRAMES):
        sim.solver.advance(sim.potential, sim.pump_field, sim.reservoir,
                           sim.boundary_condition, sim.state, step, FRAME_EVERY)
        step += FRAME_EVERY
        if i >= FIRST_FRAME:
            frames.append(render(cp.asnumpy(sim.state.psi)))

    OUT.mkdir(parents=True, exist_ok=True)
    frames[-1].save(OUT / "ring_vortices.png", optimize=True)
    frames[0].save(OUT / "ring_vortices.webp", save_all=True, append_images=frames[1:],
                   duration=50, loop=0, quality=80)


if __name__ == "__main__":
    main()
