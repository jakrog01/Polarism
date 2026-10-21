"""Command-line interface for a single Polarism simulation."""
from __future__ import annotations

import tyro
import sys
import yaml

from polarism import Config, SimulationController, Sweep
from polarism.compute_engine import compute_engine
from polarism.config.validation import validate_config
from polarism.plugins import load_plugins


def main() -> None:
    """Run one simulation from command-line configuration options."""
    load_plugins()
    args = sys.argv[1:]
    sweep_path = None
    if "--sweep" in args:
        index = args.index("--sweep")
        if index + 1 >= len(args):
            raise ValueError("--sweep requires a YAML path")
        sweep_path = args[index + 1]
        del args[index:index + 2]
    cfg = tyro.cli(Config, args=args)
    compute_engine.configure(cfg.compute_engine)
    if sweep_path is None:
        SimulationController(cfg).run()
    else:
        with open(sweep_path, encoding="utf-8") as handle:
            sweep_config = yaml.safe_load(handle) or {}
        if not isinstance(sweep_config, dict) or "parameters" not in sweep_config:
            raise ValueError("sweep YAML must contain a parameters mapping")
        settings = sweep_config.get("sweep", {})
        if not isinstance(settings, dict):
            raise ValueError("sweep YAML 'sweep' settings must be a mapping")
        for name, value in settings.items():
            if not hasattr(cfg.sweep, name):
                raise ValueError(f"Unknown sweep setting {name!r}")
            setattr(cfg.sweep, name, value)
        validate_config(cfg)
        Sweep(cfg, sweep_config["parameters"], mode=sweep_config.get("mode", "product")).run()
