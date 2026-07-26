"""End-to-end dynamic SNN pipeline."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mnist_digits_polariton_snn_dynamic.config.loader import (
    build_encoder,
    build_geometry,
    load_polarism_config,
    load_snn_dynamic_config,
)
from mnist_digits_polariton_snn_dynamic.encoding.downsample import load_and_downsample
from mnist_digits_polariton_snn_dynamic.io.atomic import atomic_write_json
from mnist_digits_polariton_snn_dynamic.readout.classifier import ClassificationReport, train_and_evaluate
from mnist_digits_polariton_snn_dynamic.readout.features import collect_features
from mnist_digits_polariton_snn_dynamic.readout.reporting import (
    plot_confusion_aggregate,
    plot_per_class_confusion_rows,
    plot_per_class_metrics,
)
from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources


def run_pipeline(
    config_path: str,
    output_dir: str | None = None,
    power_max_override: float | None = None,
) -> ClassificationReport:
    """Run the full dynamic SNN pipeline.

    Parameters
    ----------
    config_path
        Path to ``src/mnist_digits_polariton_snn_dynamic/config.yaml``.
    output_dir
        Optional override for ``SNNDynamicConfig.output_dir``. Intended for
        HPC submitters that route artifacts into a per-run directory.

    Returns
    -------
    ClassificationReport
        Final linear-readout classification metrics.
    """
    snn_cfg = load_snn_dynamic_config(config_path)
    if power_max_override is not None:
        snn_cfg = dataclasses.replace(
            snn_cfg,
            encoding=dataclasses.replace(
                snn_cfg.encoding,
                power_max=float(power_max_override),
            ),
        )
    effective_output_dir = output_dir if output_dir is not None else snn_cfg.output_dir
    polarism_cfg = load_polarism_config(snn_cfg.polarism_config_path)
    if polarism_cfg.solver.precision != "double":
        raise ValueError(
            "mnist_digits_polariton_snn_dynamic requires solver.precision == 'double', "
            f"got {polarism_cfg.solver.precision!r}"
        )
    if polarism_cfg.laser.laser_type != "pulse-gaussian":
        raise ValueError(
            "mnist_digits_polariton_snn_dynamic requires laser.laser_type == 'pulse-gaussian', "
            f"got {polarism_cfg.laser.laser_type!r}"
        )
    geometry = build_geometry(snn_cfg.geometry)
    encoder = build_encoder(snn_cfg.encoding)
    if encoder.n_spots != geometry.n_spots:
        raise ValueError(
            f"encoder.n_spots={encoder.n_spots} does not match "
            f"geometry.n_spots={geometry.n_spots}"
        )
    input_shape = encoder.input_shape()
    if len(input_shape) != 2 or input_shape[0] != input_shape[1]:
        raise ValueError(
            f"MNIST downsampling requires a square encoder input, got {input_shape}"
        )
    images, labels = load_and_downsample(
        snn_cfg.data.path,
        snn_cfg.data.n_samples,
        snn_cfg.data.seed,
        target_side=input_shape[0],
    )
    powers = encoder.encode(images)
    resources = SharedSimResources(
        polarism_cfg,
        geometry,
        snn_cfg.pulse,
        mask_radius_um=snn_cfg.readout.mask_radius_um,
    )
    bundle = collect_features(
        resources,
        powers,
        labels,
        readout=snn_cfg.readout,
    )
    report = train_and_evaluate(
        bundle,
        seed=snn_cfg.data.seed,
        test_fraction=snn_cfg.classifier.test_fraction,
        C=snn_cfg.classifier.C,
        max_iter=snn_cfg.classifier.max_iter,
    )
    out_dir = Path(effective_output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = _build_metadata(
        config_path,
        snn_cfg.polarism_config_path,
        geometry.describe(),
        encoder.describe(),
        {
            "mask_radius_um": float(snn_cfg.readout.mask_radius_um),
            "power_max_override": (
                float(power_max_override) if power_max_override is not None else None
            ),
        },
    )
    _atomic_write_npy(out_dir / "features.npy", bundle.features)
    _atomic_write_npy(out_dir / "labels.npy", bundle.labels)
    _atomic_write_npy(out_dir / "input_images.npy", images)
    _atomic_write_npy(out_dir / "encoded_powers.npy", powers)
    _atomic_write_npy(out_dir / "traces_psi.npy", bundle.traces_psi)
    if bundle.traces_n_active is not None:
        _atomic_write_npy(out_dir / "traces_nA.npy", bundle.traces_n_active)
    if bundle.traces_n_inactive is not None:
        _atomic_write_npy(out_dir / "traces_nI.npy", bundle.traces_n_inactive)
    _atomic_write_npy(out_dir / "trace_times_ps.npy", bundle.trace_times_ps)
    atomic_write_json(str(out_dir / "report.json"), dataclasses.asdict(report))
    plot_confusion_aggregate(
        report.confusion_test,
        report.confusion_test_normalized,
        out_dir / "confusion_aggregate.png",
    )
    plot_per_class_metrics(
        report.per_class_precision_test,
        report.per_class_recall_test,
        report.per_class_f1_test,
        out_dir / "per_class_metrics.png",
        overall_accuracy=report.accuracy_test,
    )
    plot_per_class_confusion_rows(
        report.confusion_test_normalized,
        out_dir / "per_class_confusion_rows.png",
    )
    atomic_write_json(str(out_dir / "r_metadata.json"), metadata)
    atomic_write_json(
        str(out_dir / "run_metadata.json"),
        metadata,
    )
    return report


def main() -> None:
    """Run the dynamic SNN pipeline from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/mnist_digits_polariton_snn_dynamic/config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--power-max", type=float, default=None)
    args = parser.parse_args()
    report = run_pipeline(args.config, output_dir=args.output_dir, power_max_override=args.power_max)
    print(json.dumps(_json_ready(dataclasses.asdict(report)), indent=2))


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npy")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def _build_metadata(
    config_path: str,
    polarism_config_path: str,
    geometry: dict[str, Any],
    encoder: dict[str, Any],
    readout: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).expanduser().resolve()),
        "polarism_config_path": str(Path(polarism_config_path).expanduser().resolve()),
        "config_sha256": _sha256_file(config_path),
        "polarism_config_sha256": _sha256_file(polarism_config_path),
        "geometry": geometry,
        "encoder": encoder,
        "readout": readout,
        "git_sha": _git_sha(),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(Path(path).expanduser(), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
