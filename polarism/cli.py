"""Command-line interface for a single Polarism simulation."""
from __future__ import annotations

import tyro

from polarism import Config, SimulationController
from polarism.compute_engine import compute_engine


def main() -> None:
    """Run one simulation from command-line configuration options."""
    cfg = tyro.cli(Config)
    compute_engine.configure(cfg.compute_engine)
    SimulationController(cfg).run()
