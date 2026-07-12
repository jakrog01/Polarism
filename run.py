"""Command-line entry point for running one simulation."""
from __future__ import annotations

import sys
from pathlib import Path

import tyro

import polarism as ps
from polarism.compute_engine import compute_engine


def main() -> None:
    """Run the command-line entry point."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--snn-config":
        root = Path(__file__).resolve().parent
        sys.path[:0] = [
            str(root / "src" / "snn_dynamic"),
            str(root / "src" / "mnist_common"),
        ]
        from snn_dynamic.pipeline import main as snn_main

        sys.argv = [sys.argv[0], "--config", sys.argv[2], *sys.argv[3:]]
        snn_main()
        return
    cfg = tyro.cli(ps.Config)
    compute_engine.configure(cfg.compute_engine)
    controller = ps.SimulationController(cfg)
    controller.run()


if __name__ == "__main__":
    main()
