"""CPU Stage 1: prepare MNIST reference for N images.

For each selected MNIST image:
  1. Selects and normalises the image.
  2. Encodes pixels as a temporal pulse sequence.
  3. Runs the quadratic-double reservoir ODE to obtain the target condensate trace.
  4. Saves per-image reference data.

Output structure::

  <run_dir>/reference/images/index.json
  <run_dir>/reference/images/image_000/input_image.npy
  <run_dir>/reference/images/image_000/input_image.png
  <run_dir>/reference/images/image_000/encoded_events.json
  <run_dir>/reference/images/image_000/target_trace.npz
  ...
  <run_dir>/reference/encoded_events.json     (copy of image_000, backward compat)
  <run_dir>/reference/target_trace.npz        (copy of image_000, backward compat)
  <run_dir>/reference/input_image.npy         (copy of image_000, backward compat)

Invoked as::

    python -m dot_response_fit.stages.cpu.prepare_reference \\
        --config <run_dir>/config.yaml --run-dir <run_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dot_response_fit.config.loader import (
    extract_encoding_cfg,
    extract_mnist_cfg,
    extract_reference_cfg,
    load_config,
    make_dataclass,
)
from dot_response_fit.manifest.io import atomic_write_json, set_manifest_field
from dot_response_fit.physics.reference_ode import params_from_physics, run_reference_ode
from polarism.config.simulation_parameters import PhysicsConstants
from polarism.encoder.amplitude_encoder import AmplitudeEncoder


def _load_mnist(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` from a local ``.npz`` file."""
    path = os.path.expanduser(cfg["data_path"])
    if not os.path.isfile(path):
        print(f"ERROR: MNIST data file not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(path)
    for img_key, lbl_key in (
        ("x_train", "y_train"),
        ("training_images", "training_labels"),
        ("images", "labels"),
        ("train_images", "train_labels"),
    ):
        if img_key in data and lbl_key in data:
            return data[img_key], data[lbl_key]

    print(
        f"ERROR: Could not find image/label arrays in {path}.\n"
        f"  Found keys: {list(data.keys())}",
        file=sys.stderr,
    )
    sys.exit(1)


def _select_samples(
    images: np.ndarray,
    labels: np.ndarray,
    digit_class: int | None,
    sample_index: int | None,
    sample_indices: list[int] | None,
    n_images: int,
    seed: int,
) -> list[tuple[np.ndarray, int, int]]:
    """Return a list of ``(image_2d, true_class, dataset_index)`` tuples.

    Selection priority:
    1. ``sample_indices`` — use exactly these indices within the digit pool.
    2. ``sample_index``   — single-image backward-compatible mode.
    3. Random draw of ``n_images`` from the digit pool.
    """
    rng = np.random.default_rng(seed)
    pool = (
        np.where(labels == digit_class)[0]
        if digit_class is not None
        else np.arange(len(labels))
    )
    if len(pool) == 0:
        print(f"ERROR: No samples found for digit class {digit_class}.", file=sys.stderr)
        sys.exit(1)

    if sample_indices is not None:
        if any(si >= len(pool) for si in sample_indices):
            print(
                f"ERROR: sample_indices out of range for pool size {len(pool)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        dataset_indices = [int(pool[si]) for si in sample_indices]
    elif sample_index is not None:
        if sample_index >= len(pool):
            print(
                f"ERROR: sample_index={sample_index} out of range for pool size {len(pool)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        dataset_indices = [int(pool[sample_index])]
    else:
        n = min(n_images, len(pool))
        dataset_indices = [int(i) for i in rng.choice(pool, size=n, replace=False)]

    result = []
    for idx in dataset_indices:
        img = images[idx]
        if img.ndim == 3:
            img = img[:, :, 0]
        result.append((img, int(labels[idx]), int(idx)))
    return result


def _normalize_image(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float64)
    if arr.max() > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _select_pixels(
    image_norm: np.ndarray,
    max_pixels: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(flat_indices, values)`` for the brightest pixels.

    Selects the top ``max_pixels`` by intensity, then sorts by flat index
    so temporal order follows the raster scan.
    """
    flat = image_norm.flatten()
    if max_pixels is not None and len(flat) > max_pixels:
        top_k = np.argsort(flat)[::-1][:max_pixels]
        selected_idx = np.sort(top_k)
    else:
        selected_idx = np.arange(len(flat))
    return selected_idx, flat[selected_idx]


def _pixel_positions(
    pixel_flat_indices: np.ndarray,
    image_shape: tuple[int, int],
    spatial_width_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map flat pixel indices to physical (x, y) coordinates in µm.

    Image centre → (0, 0).  Rows increase downward (screen convention),
    physical y increases upward, so rows are negated.
    """
    h, w = image_shape
    rows = pixel_flat_indices // w
    cols = pixel_flat_indices % w
    pixel_size_x = spatial_width_um / w
    pixel_size_y = spatial_width_um / h
    x_positions = (cols - w / 2.0 + 0.5) * pixel_size_x
    y_positions = -(rows - h / 2.0 + 0.5) * pixel_size_y
    return x_positions, y_positions


def _build_encoding(
    pixel_values: np.ndarray,
    pixel_flat_indices: np.ndarray,
    image_shape: tuple[int, int],
    enc_cfg: dict,
    global_cfg: dict,
    spatial_width_um: float,
) -> dict:
    min_amp = float(enc_cfg["min_amp"])
    max_amp = float(enc_cfg["max_amp"])
    fwhm = float(enc_cfg["pulse_width_fwhm"])
    separation = float(enc_cfg["separation"])
    cutoff_sigma = float(global_cfg.get("laser_defaults", {}).get("cutoff_sigma", 3.0))
    laser_defaults = global_cfg.get("laser_defaults", {})
    laser_x0 = float(laser_defaults.get("x0", 0.0))
    laser_y0 = float(laser_defaults.get("y0", 0.0))

    sigma_t = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    phase = cutoff_sigma * sigma_t

    encoder = AmplitudeEncoder(
        max_amp=max_amp,
        pulse_width=fwhm,
        separation=separation,
        min_amp=min_amp,
    )
    amps, _ = encoder.encode(pixel_values)

    n_pixels = len(pixel_values)
    centers_ode = phase + np.arange(n_pixels, dtype=np.float64) * separation
    delays = np.arange(n_pixels, dtype=np.float64) * separation
    total_sim_time = centers_ode[-1] + phase + 200.0

    x_pos, y_pos = _pixel_positions(pixel_flat_indices, image_shape, spatial_width_um)
    rows = pixel_flat_indices // image_shape[1]
    cols = pixel_flat_indices % image_shape[1]

    return {
        "n_pixels": n_pixels,
        "image_shape": list(image_shape),
        "amplitudes": amps.tolist(),
        "delays": delays.tolist(),
        "centers_ode": centers_ode.tolist(),
        "x_positions": x_pos.tolist(),
        "y_positions": y_pos.tolist(),
        "laser_x0": laser_x0,
        "laser_y0": laser_y0,
        "spatial_encoding": "fixed-dot",
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        "sigma_time": sigma_t,
        "cutoff_sigma": cutoff_sigma,
        "phase": phase,
        "separation": separation,
        "pulse_width_fwhm": fwhm,
        "min_amp": min_amp,
        "max_amp": max_amp,
        "spatial_width_um": spatial_width_um,
        "pixel_flat_indices": pixel_flat_indices.tolist(),
        "total_sim_time": total_sim_time,
    }


def _save_input_image_png(image_norm: np.ndarray, out_path: str) -> None:
    """Save the normalised MNIST image as a clean greyscale PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(image_norm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _process_image(
    img_2d: np.ndarray,
    true_class: int,
    dataset_idx: int,
    image_id: str,
    image_dir: str,
    enc_cfg: dict,
    global_cfg: dict,
    ref_cfg: dict,
    max_pixels: int | None,
    spatial_width_um: float,
    physics_params: dict,
    n_points: int,
    rtol: float,
    atol: float,
    seed: int,
) -> dict[str, Any]:
    """Process one MNIST image and save all reference artifacts.

    Returns a metadata dict for the index file.
    """
    os.makedirs(image_dir, exist_ok=True)

    image_norm = _normalize_image(img_2d)
    pixel_flat_indices, pixel_values = _select_pixels(image_norm, max_pixels)
    n_pixels = len(pixel_flat_indices)

    nc_source = float(ref_cfg.get("nc_source", 1.0e-6))
    params = {**physics_params, "nc_source": nc_source}

    events = _build_encoding(
        pixel_values, pixel_flat_indices, image_norm.shape,
        enc_cfg, global_cfg, spatial_width_um,
    )
    events["digit_class"] = true_class
    events["dataset_index"] = dataset_idx
    events["seed"] = seed
    events["reference_nc_source"] = nc_source
    events["reference_initial_state_nR_nI_nC"] = [0.0, 0.0, nc_source]

    amps = np.array(events["amplitudes"])
    centers_ode = np.array(events["centers_ode"])
    sigma_t = float(events["sigma_time"])
    t_end = float(events["total_sim_time"])

    t_trace, nC_trace, pump_trace = run_reference_ode(
        amps, centers_ode, sigma_t, params, t_end, n_points, rtol, atol
    )

    npy_path = os.path.join(image_dir, "input_image.npy")
    png_path = os.path.join(image_dir, "input_image.png")
    events_path = os.path.join(image_dir, "encoded_events.json")
    trace_path = os.path.join(image_dir, "target_trace.npz")

    np.save(npy_path, image_norm)
    _save_input_image_png(image_norm, png_path)
    atomic_write_json(events_path, events)
    np.savez_compressed(trace_path, time=t_trace, nC=nC_trace, pump=pump_trace)

    return {
        "image_id": image_id,
        "dataset_index": dataset_idx,
        "digit_class": true_class,
        "n_pixels": n_pixels,
        "total_sim_time": t_end,
        "nC_max": float(nC_trace.max()),
        "paths": {
            "input_image_npy": os.path.join("reference", "images", image_id, "input_image.npy"),
            "input_image_png": os.path.join("reference", "images", image_id, "input_image.png"),
            "encoded_events": os.path.join("reference", "images", image_id, "encoded_events.json"),
            "target_trace": os.path.join("reference", "images", image_id, "target_trace.npz"),
        },
    }


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="CPU Stage 1: prepare MNIST reference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    global_cfg = cfg["global"]
    mnist_cfg = extract_mnist_cfg(cfg)
    enc_cfg = extract_encoding_cfg(cfg)
    ref_cfg = extract_reference_cfg(cfg)

    print("=" * 60)
    print(" CPU Stage 1: Prepare MNIST Reference")
    print("=" * 60)
    print(f"  Run dir : {run_dir}")

    images, labels = _load_mnist(mnist_cfg)
    print(f"  MNIST   : {images.shape[0]} samples loaded")

    digit_class = mnist_cfg.get("digit_class")
    sample_index = mnist_cfg.get("sample_index")
    sample_indices = mnist_cfg.get("sample_indices")
    n_images = int(mnist_cfg.get("n_images", 1))
    seed = int(mnist_cfg.get("seed", 42))
    max_pixels = mnist_cfg.get("max_pixels")
    if max_pixels is not None:
        max_pixels = int(max_pixels)
    spatial_width_um = float(mnist_cfg.get("spatial_width_um", 80.0))

    samples = _select_samples(
        images, labels, digit_class, sample_index, sample_indices, n_images, seed
    )
    if not samples:
        print(
            "ERROR: no images selected. Check mnist.sample_indices (must be non-empty), "
            "mnist.sample_index, or mnist.n_images in config.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Selected: {len(samples)} image(s)")

    physics = make_dataclass(PhysicsConstants, global_cfg.get("physics", {}))
    physics_params = params_from_physics(physics)

    n_points = int(ref_cfg.get("n_points", 2000))
    rtol = float(ref_cfg.get("ode_solver_rtol", 1e-6))
    atol = float(ref_cfg.get("ode_solver_atol", 1e-7))

    ref_dir = os.path.join(run_dir, "reference")
    images_dir = os.path.join(ref_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    index: list[dict[str, Any]] = []
    for i, (img_2d, true_class, dataset_idx) in enumerate(samples):
        image_id = f"image_{i:03d}"
        image_dir = os.path.join(images_dir, image_id)
        print(
            f"  [{i + 1}/{len(samples)}] {image_id}  "
            f"dataset_idx={dataset_idx}  class={true_class}",
            flush=True,
        )

        entry = _process_image(
            img_2d=img_2d,
            true_class=true_class,
            dataset_idx=dataset_idx,
            image_id=image_id,
            image_dir=image_dir,
            enc_cfg=enc_cfg,
            global_cfg=global_cfg,
            ref_cfg=ref_cfg,
            max_pixels=max_pixels,
            spatial_width_um=spatial_width_um,
            physics_params=physics_params,
            n_points=n_points,
            rtol=rtol,
            atol=atol,
            seed=seed,
        )
        index.append(entry)
        print(
            f"    n_pixels={entry['n_pixels']}  "
            f"t_end={entry['total_sim_time']:.1f} ps  "
            f"nC_max={entry['nC_max']:.4e}"
        )

    index_path = os.path.join(images_dir, "index.json")
    atomic_write_json(index_path, index)
    print(f"\n  Index   : {index_path}  ({len(index)} images)")

    first = index[0]
    first_dir = os.path.join(images_dir, first["image_id"])
    for fname in ("input_image.npy", "encoded_events.json", "target_trace.npz"):
        src = os.path.join(first_dir, fname)
        dst = os.path.join(ref_dir, fname)
        import shutil
        shutil.copy2(src, dst)
    print("  Backward-compat copies written to reference/ root (image_000).")

    try:
        set_manifest_field(run_dir, "prepare_reference_complete", True)
        set_manifest_field(run_dir, "n_images", len(index))
        set_manifest_field(run_dir, "image_index_path", index_path)
    except Exception as exc:
        print(f"  WARNING: manifest update failed: {exc}", file=sys.stderr)

    print("\n  Stage 1 complete.")


if __name__ == "__main__":
    main()
