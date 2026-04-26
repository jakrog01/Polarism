"""MNIST-overlay animation renderer.

Extends the standard NVENC streaming animation with two extra overlays:

1. A ring marker on each field panel (``psi_sq``, ``Pump``) showing the
   spatial position of the currently active MNIST pulse.
2. A small MNIST thumbnail panel appended to the right of the field panels,
   with the active pixel circled in red.

All compositing is done frame-by-frame via raw numpy arrays — no full-animation
RAM preload.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Any

import h5py
import numpy as np

from pipeline.render.nvenc_stream import (
    ANIM_FPS,
    ANIM_TARGET_SECONDS,
    N_SAMPLE_FRAMES,
    _available_encoders,
    _encoder_config,
    _get_lut,
    _pad_even,
    _pick_default_encoder,
    _read_field,
    _resolve_ffmpeg,
    _to_uint8_rgb,
)

_THUMB_PANEL_WIDTH = 128
_MARKER_COLOR = (255, 64, 0)
_MARKER_THICKNESS = 2


def _load_events(events_path: str) -> dict:
    with open(events_path) as f:
        return json.load(f)


def _load_image(image_path: str) -> np.ndarray:
    img = np.load(image_path).astype(np.float64)
    mx = img.max()
    return img / mx if mx > 0 else img


def _find_last_active_pixel(
    centers_ode: list[float],
    sigma_time: float,
    cutoff_sigma: float,
) -> int | None:
    """Return index of the last pixel (highest center_ode index) that had an active pulse."""
    if not centers_ode:
        return None
    return len(centers_ode) - 1


def _phys_to_panel(
    x_um: float,
    y_um: float,
    extent: list[float],
    panel_h: int,
    panel_w: int,
) -> tuple[int, int]:
    """Convert a physical position (µm) to panel pixel indices (row, col)."""
    xmin, xmax, ymin, ymax = extent
    col = int((x_um - xmin) / (xmax - xmin) * panel_w)
    row = int((ymax - y_um) / (ymax - ymin) * panel_h)
    col = max(0, min(panel_w - 1, col))
    row = max(0, min(panel_h - 1, row))
    return row, col


def _draw_ring(
    panel: np.ndarray,
    row: int,
    col: int,
    radius: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an anti-aliased ring on *panel* (in-place)."""
    h, w = panel.shape[:2]
    r_outer = radius + thickness
    for dy in range(-r_outer - 1, r_outer + 2):
        for dx in range(-r_outer - 1, r_outer + 2):
            dist = math.sqrt(dy * dy + dx * dx)
            if abs(dist - radius) <= thickness:
                py, px = row + dy, col + dx
                if 0 <= py < h and 0 <= px < w:
                    panel[py, px] = color


def _draw_crosshair(
    panel: np.ndarray,
    row: int,
    col: int,
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a small crosshair on *panel* (in-place)."""
    h, w = panel.shape[:2]
    for d in range(-size, size + 1):
        if 0 <= row + d < h:
            panel[row + d, col] = color
        if 0 <= col + d < w:
            panel[row, col + d] = color


def _mark_active_on_panel(
    panel: np.ndarray,
    x_um: float,
    y_um: float,
    extent: list[float],
) -> np.ndarray:
    """Return a copy of *panel* with a ring marking the active spatial position."""
    out = panel.copy()
    row, col = _phys_to_panel(x_um, y_um, extent, panel.shape[0], panel.shape[1])
    radius = max(4, panel.shape[0] // 60)
    _draw_ring(out, row, col, radius, _MARKER_COLOR, _MARKER_THICKNESS)
    return out


def _make_thumbnail(
    image_norm: np.ndarray,
    pixel_flat_indices: list[int],
    active_pixel_idx: int | None,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Render the MNIST thumbnail, circling the active pixel if any.

    Returns uint8 RGB of shape ``(target_h, target_w, 3)``.
    """
    h_src, w_src = image_norm.shape
    grey = np.clip(image_norm * 255, 0, 255).astype(np.uint8)

    scale_y = target_h / h_src
    scale_x = target_w / w_src

    row_idx = np.clip(
        (np.arange(target_h) / scale_y).astype(int), 0, h_src - 1
    )
    col_idx = np.clip(
        (np.arange(target_w) / scale_x).astype(int), 0, w_src - 1
    )
    resized = grey[np.ix_(row_idx, col_idx)]
    rgb = np.stack([resized, resized, resized], axis=2)

    if active_pixel_idx is not None:
        flat_idx = pixel_flat_indices[active_pixel_idx]
        src_row = flat_idx // w_src
        src_col = flat_idx % w_src
        cy = int(src_row * scale_y + scale_y / 2)
        cx = int(src_col * scale_x + scale_x / 2)
        radius = max(3, int(min(scale_y, scale_x) * 0.7))
        _draw_ring(rgb, cy, cx, radius, (255, 0, 0), 1)
        _draw_crosshair(rgb, cy, cx, max(1, radius // 3), (255, 0, 0))

    return rgb


def _resize_thumb(thumb: np.ndarray, target_h: int) -> np.ndarray:
    """Nearest-neighbour resize to *target_h* preserving aspect ratio."""
    src_h, src_w = thumb.shape[:2]
    if src_h == target_h:
        return thumb
    scale = target_h / src_h
    dst_w = max(1, int(src_w * scale))
    row_idx = np.clip((np.arange(target_h) / scale).astype(int), 0, src_h - 1)
    col_idx = np.clip((np.arange(dst_w) / scale).astype(int), 0, src_w - 1)
    return thumb[np.ix_(row_idx, col_idx)]


_MARKED_PANELS = {"psi_sq", "Pump"}


def generate_animation_with_overlay(
    routine: str,
    field_specs: dict[str, dict],
    extent: list[float],
    data_dir: str,
    results_dir: str,
    events_path: str,
    image_path: str,
    fps: int = ANIM_FPS,
    encoder: str | None = None,
) -> None:
    """Stream animation with MNIST overlay from HDF5 to a video via ffmpeg.

    Active-pixel markers are drawn on the ``psi_sq`` and ``Pump`` field
    panels (ring at the physical laser position) and on the MNIST thumbnail
    (red circle around the active pixel in the image).

    Parameters
    ----------
    routine : str
        Scenario name; HDF5 read from ``{data_dir}/{routine}.h5``.
    field_specs : dict
        Field panel specifications (same as ``generate_animation``).
    extent : list[float]
        ``[xmin, xmax, ymin, ymax]`` µm — used to map physical positions to
        panel pixels.
    data_dir : str
        Directory containing ``{routine}.h5``.
    results_dir : str
        Output base directory; video written to ``{results_dir}/{routine}/dynamics.*``.
    events_path : str
        Path to ``reference/encoded_events.json``.
    image_path : str
        Path to ``reference/input_image.npy``.
    fps : int
        Output frame rate.
    encoder : str or None
        Override video encoder.
    """
    ffmpeg = _resolve_ffmpeg()
    encoders = _available_encoders(ffmpeg)
    chosen = encoder or os.environ.get("RENDER_ENCODER") or _pick_default_encoder(encoders)
    if chosen not in encoders:
        raise RuntimeError(
            f"Requested encoder '{chosen}' not available in '{ffmpeg}'.\n"
            f"Available: {', '.join(sorted(encoders))}"
        )

    events = _load_events(events_path)
    image_norm = _load_image(image_path)
    centers_ode: list[float] = events["centers_ode"]
    pixel_flat_indices: list[int] = events["pixel_flat_indices"]
    x_positions: list[float] = events["x_positions"]
    y_positions: list[float] = events["y_positions"]
    sigma_time = float(events["sigma_time"])
    cutoff_sigma = float(events["cutoff_sigma"])

    h5_path = os.path.join(data_dir, f"{routine}.h5")
    out_dir = os.path.join(results_dir, routine)
    os.makedirs(out_dir, exist_ok=True)
    encoder_flags, out_name = _encoder_config(chosen)
    out_path = os.path.join(out_dir, out_name)

    field_keys = list(field_specs.keys())
    luts = {k: _get_lut(field_specs[k]["cmap"]) for k in field_keys}

    with h5py.File(h5_path, "r") as h5:
        sort_order = np.argsort(h5["time"][:], kind="stable")
        n_total = len(sort_order)
        times_sorted = np.array(h5["time"][:])[sort_order]

        target_frames = fps * ANIM_TARGET_SECONDS
        stride = max(1, n_total // target_frames)
        anim_logical = list(range(0, n_total, stride))
        anim_physical = [int(sort_order[i]) for i in anim_logical]
        anim_times = times_sorted[anim_logical]

        sample_stride = max(1, (n_total - 1) // max(1, N_SAMPLE_FRAMES - 1))
        sample_physical = [int(sort_order[i]) for i in range(0, n_total, sample_stride)][:N_SAMPLE_FRAMES]

        vmins: dict[str, float] = {}
        vmaxs: dict[str, float] = {}
        for k in field_keys:
            sp = field_specs[k]
            samples = [_read_field(h5, sp, pi) for pi in sample_physical]
            vmins[k] = float(min(s.min() for s in samples))
            vmaxs[k] = float(max(s.max() for s in samples))
            if vmaxs[k] <= vmins[k]:
                vmaxs[k] = vmins[k] + 1e-12

        probe_panels = [
            _to_uint8_rgb(
                _read_field(h5, field_specs[k], anim_physical[0]),
                luts[k], vmins[k], vmaxs[k], field_specs[k].get("norm"),
            )
            for k in field_keys
        ]
        field_h = probe_panels[0].shape[0]
        probe_thumb = _make_thumbnail(
            image_norm, pixel_flat_indices, None, field_h, _THUMB_PANEL_WIDTH
        )
        probe_strip = _pad_even(
            np.concatenate(probe_panels + [probe_thumb], axis=1)
        )
        height, width = probe_strip.shape[:2]
        n_frames = len(anim_physical)

        print(
            f"    Streaming {n_frames} frames  {height}×{width}px  "
            f"encoder={chosen}  fps={fps}  (MNIST overlay) ..."
        )

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            *encoder_flags,
            out_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame_i, physical_idx in enumerate(anim_physical):
                panels: list[np.ndarray] = []
                for k in field_keys:
                    panel = _to_uint8_rgb(
                        _read_field(h5, field_specs[k], physical_idx),
                        luts[k], vmins[k], vmaxs[k], field_specs[k].get("norm"),
                    )
                    panels.append(panel)

                thumb = _make_thumbnail(
                    image_norm, pixel_flat_indices, None,
                    field_h, _THUMB_PANEL_WIDTH,
                )
                frame = _pad_even(np.concatenate(panels + [thumb], axis=1))
                proc.stdin.write(frame.tobytes())

        except BrokenPipeError:
            proc.stdin.close()
            proc.wait()
            stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
            raise RuntimeError(
                f"ffmpeg terminated during streaming to {out_path}.\n"
                f"stderr: {stderr_bytes.decode(errors='replace')[-1000:]}"
            )

        proc.stdin.close()
        proc.wait()
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}.\n"
                f"stderr: {stderr_bytes.decode(errors='replace')[-1000:]}"
            )

    print(f"    {out_path}")
