"""Run threshold calibration for one campaign scenario."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from mnist_digits_polariton_snn_dynamic.simulation.calibration import calibrate_threshold
from mnist_digits_polariton_snn_dynamic.scenarios.stage_meta import (
    find_scenario as _find_scenario,
    load_yaml as _load_yaml,
    write_stage_meta as _write_stage_meta,
)


def main() -> None:
    """Run one scenario calibration selected from a campaign manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaign-output-dir", required=True)
    parser.add_argument("--n-points", type=int, default=17)
    parser.add_argument("--condensation-ratio", type=float, default=10.0)
    parser.add_argument("--full-lattice-power-margin", type=float, default=1.05)
    parser.add_argument("--single-spot-safety-fraction", type=float, default=0.95)
    parser.add_argument("--realistic-safety-fraction", type=float, default=0.95)
    parser.add_argument("--guard-n-points", type=int, default=10)
    args = parser.parse_args()
    run_calibration(
        args.manifest,
        args.scenario_id,
        args.campaign_output_dir,
        n_points=args.n_points,
        condensation_ratio=args.condensation_ratio,
        full_lattice_power_margin=args.full_lattice_power_margin,
        single_spot_safety_fraction=args.single_spot_safety_fraction,
        guard_n_points=args.guard_n_points,
        realistic_safety_fraction=args.realistic_safety_fraction,
    )


def run_calibration(
    manifest_path: str,
    scenario_id: str,
    campaign_output_dir: str,
    *,
    n_points: int,
    condensation_ratio: float,
    full_lattice_power_margin: float,
    single_spot_safety_fraction: float,
    guard_n_points: int,
    realistic_safety_fraction: float = 0.95,
) -> None:
    """Run threshold calibration for one scenario from a manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    manifest_data = _load_yaml(manifest)
    scenario = _find_scenario(manifest_data, scenario_id)
    config = (manifest.parent / str(scenario["config"])).resolve()
    output_dir = Path(campaign_output_dir).expanduser().resolve() / scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    error: str | None = None
    try:
        calibrate_threshold(
            str(config),
            scenario_id,
            str(output_dir),
            n_points=n_points,
            condensation_ratio=condensation_ratio,
            full_lattice_power_margin=full_lattice_power_margin,
            single_spot_safety_fraction=single_spot_safety_fraction,
            guard_n_points=guard_n_points,
            realistic_safety_fraction=realistic_safety_fraction,
        )
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        stop = datetime.now(timezone.utc)
        _write_stage_meta(
            output_dir / "calibration_meta.json",
            scenario_id,
            config,
            manifest,
            "calibrate",
            start,
            stop,
            error,
        )
if __name__ == "__main__":
    main()
