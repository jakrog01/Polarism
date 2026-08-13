"""Generate presentation figures from an existing dynamic-SNN run directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from mnist_digits_polariton_snn_dynamic.encoding.base import LinearPixelEncoder
from mnist_digits_polariton_snn_dynamic.simulation.geometry import GridLatticeGeometry


def main() -> None:
    """Generate all requested presentation snapshot figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Circle

    from mnist_digits_polariton_snn_dynamic.readout.reporting import plot_field_snapshots_grid

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--meta-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--lattice-sample-idx", type=int, default=0)
    parser.add_argument("--field-snapshots-npz", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    meta_dir = Path(args.meta_dir).expanduser().resolve() if args.meta_dir else run_dir.parent
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = _parse_classes(args.classes)
    metadata = _load_metadata(run_dir, meta_dir)
    geometry, encoder, mask_radius_um = _metadata_sections(metadata)
    lattice = GridLatticeGeometry(
        n_side=int(geometry["n_side"]),
        pitch_um=float(geometry["pitch_um"]),
        sigma_space_um=float(geometry["sigma_space_um"]),
        center_x_um=float(geometry["center_x_um"]),
        center_y_um=float(geometry["center_y_um"]),
    )
    pixel_encoder = LinearPixelEncoder(
        n_side=int(encoder["n_side"]),
        power_min=float(encoder["power_min"]),
        power_max=float(encoder["power_max"]),
    )
    images, powers, labels, traces_psi, traces_n_active, traces_n_inactive, trace_times = _load_run_arrays(
        run_dir
    )
    _validate_run_arrays(
        images,
        powers,
        labels,
        traces_psi,
        traces_n_active,
        traces_n_inactive,
        trace_times,
        lattice.n_spots,
        pixel_encoder.n_side,
    )
    if not 0 <= args.lattice_sample_idx < labels.size:
        raise ValueError(
            f"lattice-sample-idx must be in [0, {labels.size}), got {args.lattice_sample_idx}"
        )
    representative = _representatives(labels)
    _plot_lattice_layout(
        plt,
        Circle,
        Normalize,
        lattice,
        powers[args.lattice_sample_idx],
        pixel_encoder.power_max,
        mask_radius_um,
        out_dir / "lattice_layout.png",
    )
    _plot_digit_to_pump_grid(
        plt,
        images,
        powers,
        representative,
        pixel_encoder.n_side,
        pixel_encoder.power_max,
        out_dir / "digit_to_pump_grid.png",
    )
    for class_id in classes:
        sample_index = representative[class_id]
        _plot_traces_per_spot(
            plt,
            trace_times,
            traces_psi[sample_index],
            traces_n_active[sample_index],
            traces_n_inactive[sample_index],
            powers[sample_index],
            class_id,
            sample_index,
            pixel_encoder.n_side,
            out_dir / f"traces_per_spot_class{class_id}.png",
        )
    for class_id, sample_index in representative.items():
        _plot_overview(
            plt,
            Circle,
            Normalize,
            images[sample_index],
            powers[sample_index],
            lattice,
            pixel_encoder.n_side,
            pixel_encoder.power_max,
            out_dir / f"overview_sample{sample_index}.png",
        )
    if args.field_snapshots_npz:
        with np.load(Path(args.field_snapshots_npz).expanduser(), allow_pickle=False) as snapshot_data:
            plot_field_snapshots_grid(
                snapshot_data["rho"],
                snapshot_data["times_ps"],
                snapshot_data["sample_indices"],
                snapshot_data["labels"],
                snapshot_data["x_um"],
                snapshot_data["y_um"],
                out_dir / "field_snapshots_grid.png",
                lattice.positions_um,
            )


def _parse_classes(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value < 0 or value > 9 for value in values):
        raise ValueError("classes must be a non-empty comma-separated subset of 0..9")
    if len(set(values)) != len(values):
        raise ValueError("classes must not contain duplicates")
    return values


def _load_metadata(run_dir: Path, meta_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if isinstance(data, dict) and {"geometry", "encoder", "readout"} <= set(data):
            return data
    config_path = meta_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Could not load geometry metadata from run_metadata.json or config.yaml"
        )
    from mnist_digits_polariton_snn_dynamic.config.loader import load_snn_dynamic_config

    config = load_snn_dynamic_config(str(config_path))
    return {
        "geometry": {
            "n_side": config.geometry.n_side,
            "pitch_um": config.geometry.pitch_um,
            "sigma_space_um": config.geometry.sigma_space_um,
            "center_x_um": config.geometry.center_x_um,
            "center_y_um": config.geometry.center_y_um,
        },
        "encoder": {
            "n_side": config.encoding.n_side,
            "power_min": config.encoding.power_min,
            "power_max": config.encoding.power_max,
        },
        "readout": {"mask_radius_um": config.readout.mask_radius_um},
    }


def _metadata_sections(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float]:
    geometry = metadata["geometry"]
    encoder = metadata["encoder"]
    readout = metadata["readout"]
    if not isinstance(geometry, dict) or not isinstance(encoder, dict) or not isinstance(readout, dict):
        raise ValueError("run metadata geometry, encoder, and readout must be mappings")
    return geometry, encoder, float(readout["mask_radius_um"])


def _load_run_arrays(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "input_images": "input_images.npy",
        "encoded_powers": "encoded_powers.npy",
        "labels": "labels.npy",
        "traces_psi": "traces_psi.npy",
        "traces_nA": "traces_nA.npy",
        "traces_nI": "traces_nI.npy",
        "trace_times_ps": "trace_times_ps.npy",
    }
    arrays = {
        name: np.load(run_dir / filename, mmap_mode="r", allow_pickle=False)
        for name, filename in required.items()
    }
    return (
        np.asarray(arrays["input_images"], dtype=np.float64),
        np.asarray(arrays["encoded_powers"], dtype=np.float64),
        np.asarray(arrays["labels"], dtype=np.int64),
        np.asarray(arrays["traces_psi"], dtype=np.float64),
        np.asarray(arrays["traces_nA"], dtype=np.float64),
        np.asarray(arrays["traces_nI"], dtype=np.float64),
        np.asarray(arrays["trace_times_ps"], dtype=np.float64),
    )


def _validate_run_arrays(
    images: np.ndarray,
    powers: np.ndarray,
    labels: np.ndarray,
    traces_psi: np.ndarray,
    traces_n_active: np.ndarray,
    traces_n_inactive: np.ndarray,
    trace_times: np.ndarray,
    n_spots: int,
    n_side: int,
) -> None:
    sample_count = labels.size
    expected_channels = n_spots + 1
    if images.shape != (sample_count, n_side, n_side):
        raise ValueError(f"input_images shape must be (N, {n_side}, {n_side}), got {images.shape}")
    if powers.shape != (sample_count, n_spots):
        raise ValueError(f"encoded_powers shape must be (N, {n_spots}), got {powers.shape}")
    expected_traces = (sample_count, trace_times.size, expected_channels)
    if any(trace.shape != expected_traces for trace in (traces_psi, traces_n_active, traces_n_inactive)):
        raise ValueError(f"trace arrays must have shape {expected_traces}")


def _representatives(labels: np.ndarray) -> dict[int, int]:
    representatives: dict[int, int] = {}
    for class_id in range(10):
        matches = np.flatnonzero(labels == class_id)
        if matches.size == 0:
            raise ValueError(f"No sample found for class {class_id}")
        representatives[class_id] = int(matches[0])
    return representatives


def _plot_lattice_layout(
    plt: Any,
    circle: Any,
    normalize: Any,
    lattice: GridLatticeGeometry,
    powers: np.ndarray,
    power_max: float,
    mask_radius_um: float,
    out_path: Path,
) -> None:
    positions = lattice.positions_um
    norm = normalize(vmin=0.0, vmax=float(power_max))
    cmap = plt.get_cmap("inferno")
    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)
    for position, power in zip(positions, powers, strict=True):
        ax.add_patch(circle(tuple(position), lattice.sigma_space_um, facecolor=cmap(norm(power)), edgecolor="black", alpha=0.85))
        ax.add_patch(circle(tuple(position), np.sqrt(2.0 * np.log(2.0)) * lattice.sigma_space_um, fill=False, linestyle="--", edgecolor="black", linewidth=0.7))
        ax.add_patch(circle(tuple(position), mask_radius_um, fill=False, edgecolor="cyan", linewidth=0.7, alpha=0.8))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="pump power [a.u.]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Grid lattice pump layout")
    ax.text(
        0.02,
        0.98,
        f"pitch={lattice.pitch_um:g} um, sigma={lattice.sigma_space_um:g} um, N={lattice.n_spots}, mask_radius={mask_radius_um:g} um",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.autoscale_view()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_digit_to_pump_grid(
    plt: Any,
    images: np.ndarray,
    powers: np.ndarray,
    representative: dict[int, int],
    n_side: int,
    power_max: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 10, figsize=(20.0, 5.0), constrained_layout=True)
    heatmap = None
    for class_id in range(10):
        sample_index = representative[class_id]
        axes[0, class_id].imshow(images[sample_index], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, class_id].set_title(f"digit={class_id}")
        axes[0, class_id].set_axis_off()
        heatmap = axes[1, class_id].imshow(
            powers[sample_index].reshape(n_side, n_side),
            cmap="inferno",
            vmin=0.0,
            vmax=power_max,
        )
        axes[1, class_id].set_axis_off()
    if heatmap is None:
        raise RuntimeError("No digit heatmaps available")
    fig.colorbar(heatmap, ax=axes[1].tolist(), label="pump power [a.u.]")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traces_per_spot(
    plt: Any,
    times_ps: np.ndarray,
    psi: np.ndarray,
    n_active: np.ndarray,
    n_inactive: np.ndarray,
    powers: np.ndarray,
    class_id: int,
    sample_index: int,
    n_side: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(n_side, n_side, figsize=(10.0, 10.0), constrained_layout=True, sharex=True)
    for spot_index, ax in enumerate(axes.ravel()):
        ax.plot(times_ps, psi[:, spot_index], color="tab:blue", linewidth=0.7)
        ax.plot(times_ps, n_active[:, spot_index], color="tab:orange", linewidth=0.7)
        ax.plot(times_ps, n_inactive[:, spot_index], color="tab:green", linewidth=0.7)
        ax.text(
            0.02,
            0.98,
            f"{spot_index}\n{powers[spot_index]:.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        ax.tick_params(axis="y", labelleft=False)
        ax.grid(alpha=0.15)
    for ax in axes[-1, :]:
        ax.set_xlabel("t [ps]")
    fig.suptitle(f"Class {class_id}, sample idx={sample_index}, label={class_id}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(
    plt: Any,
    circle: Any,
    normalize: Any,
    image: np.ndarray,
    powers: np.ndarray,
    lattice: GridLatticeGeometry,
    n_side: int,
    power_max: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 6.0), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input digit")
    axes[0].set_axis_off()
    heatmap = axes[1].imshow(
        powers.reshape(n_side, n_side), cmap="inferno", vmin=0.0, vmax=power_max
    )
    axes[1].set_title("Encoded pump")
    axes[1].set_axis_off()
    fig.colorbar(heatmap, ax=axes[1], label="pump power [a.u.]")
    norm = normalize(vmin=0.0, vmax=float(power_max))
    cmap = plt.get_cmap("inferno")
    for position, power in zip(lattice.positions_um, powers, strict=True):
        axes[2].add_patch(circle(tuple(position), lattice.sigma_space_um, facecolor=cmap(norm(power)), edgecolor="black"))
    axes[2].set_title("Lattice pump")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x [um]")
    axes[2].set_ylabel("y [um]")
    axes[2].autoscale_view()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
