"""Command-line entry point for running one simulation."""
from __future__ import annotations

import tyro

import polarism as ps
from polarism.compute_engine import compute_engine

def main():
    """Run the command-line entry point."""
    cfg = tyro.cli(ps.Config)
    compute_engine.configure(cfg.compute_engine)
    controller = ps.SimulationController(cfg)
    controller.run()


if __name__ == "__main__":
    main()
