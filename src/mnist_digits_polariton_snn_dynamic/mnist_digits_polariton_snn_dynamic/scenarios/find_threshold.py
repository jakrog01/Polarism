"""CLI entry point for the CPU analytic threshold stage."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from mnist_digits_polariton_snn_dynamic.config.loader import load_snn_dynamic_config
from mnist_digits_polariton_snn_dynamic.scenarios.stage_meta import find_scenario, load_yaml, write_stage_meta
from mnist_digits_polariton_snn_dynamic.simulation.spike_threshold import SpikeThresholdSettings, find_spike_threshold


def main() -> None:
    """Run a threshold scan for a manifest scenario or a standalone config."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--config")
    parser.add_argument("--scenario-id")
    parser.add_argument("--campaign-output-dir")
    parser.add_argument("--output-dir")
    for name, kind in (("p-min", float), ("p-max", float), ("n-points", int), ("window-start-ps", float), ("window-end-ps", float), ("dt-eval-ps", float), ("hysteresis-rel", float), ("min-above-ps", float), ("edge-tol-rel", float), ("spontaneous-source", float)):
        parser.add_argument(f"--{name}", type=kind, default=None)
    parser.add_argument("--scale", choices=("log", "linear"), default=None)
    parser.add_argument("--model", choices=("pump_only", "coupled"), default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.manifest and (not args.scenario_id or not args.campaign_output_dir):
        parser.error("--manifest requires --scenario-id and --campaign-output-dir")
    if args.config and (args.scenario_id or args.campaign_output_dir or not args.output_dir):
        parser.error("--config requires --output-dir and cannot be combined with manifest options")
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        scenario = find_scenario(load_yaml(manifest), args.scenario_id)
        config = (manifest.parent / str(scenario["config"])).resolve()
        output = Path(args.campaign_output_dir).expanduser().resolve() / args.scenario_id
        _run(config, args.scenario_id, output, args, manifest)
    else:
        config = Path(args.config).expanduser().resolve()
        _run(config, config.stem, Path(args.output_dir).expanduser().resolve(), args, None)


def _run(config: Path, scenario_id: str, output: Path, args: argparse.Namespace, manifest: Path | None) -> None:
    base = SpikeThresholdSettings(**asdict(load_snn_dynamic_config(str(config)).threshold))
    updates = {key: value for key, value in vars(args).items() if key.replace("_", "-") in {"p-min", "p-max", "n-points", "scale", "window-start-ps", "window-end-ps", "dt-eval-ps", "hysteresis-rel", "min-above-ps", "edge-tol-rel", "model", "spontaneous-source"} and value is not None}
    settings = replace(base, **updates, make_plot=not args.no_plot)
    output.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    error: str | None = None
    try:
        find_spike_threshold(str(config), scenario_id, str(output), settings)
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        if manifest is not None:
            write_stage_meta(output / "threshold_meta.json", scenario_id, config, manifest, "threshold", start, datetime.now(timezone.utc), error)


if __name__ == "__main__":
    main()
