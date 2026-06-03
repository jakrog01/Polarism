"""Post-simulation streaming animation renderer.

Renders scenario fields from an HDF5 file to a video file via ffmpeg.
Reads one frame at a time — no whole-animation preload into RAM.

Colour limits are computed from the full recorded HDF5 trajectory before
encoding.  This avoids the common online-rendering failure mode where early
low-amplitude frames set the scale and the later condensate saturates the
movie.

Frame layout: field panels tiled horizontally into a single uint8 RGB
buffer, streamed frame-by-frame into ffmpeg stdin.

Environment:
    FFMPEG_BIN     Path to the ffmpeg binary (required if not on PATH).
    RENDER_ENCODER Override the video encoder. If unset, the renderer
                   picks the best available encoder for the local ffmpeg.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any

import h5py
import numpy as np

try:
    import matplotlib.cm as _cm

    def _get_lut(cmap_name: str) -> np.ndarray:
        cmap = _cm.get_cmap(cmap_name)
        return np.uint8(cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255)

except ImportError:
    def _get_lut(cmap_name: str) -> np.ndarray:
        raise RuntimeError("matplotlib is required for colormap LUT generation")

try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_PANEL_DISPLAY_NAMES: dict[str, str] = {
    "psi":  "|ψ|²",
    "nA":   "nA",
    "nI":   "nI",
    "Pump": "Pump",
}

_SCALEBAR_LENGTH_UM = 10.0
_FRAME_BG = (12, 12, 14)
_PANEL_OUTLINE = (230, 230, 235)
_PANEL_SUBTLE = (72, 72, 80)
_TEXT = (245, 245, 245)
_TEXT_MUTED = (188, 188, 194)
_SHADOW = (0, 0, 0)


ANIM_FPS = 8
ANIM_TARGET_SECONDS = 60
PUMP_NORM_GAMMA = 0.3


def _resolve_ffmpeg() -> str:
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg binary not found.  "
        "Set FFMPEG_BIN to the path of an ffmpeg build available on this node."
    )


def _available_encoders(ffmpeg_bin: str) -> set[str]:
    """Return the set of video encoder names exposed by *ffmpeg_bin*."""
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=False,
    )
    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def _pick_default_encoder(encoders: set[str]) -> str:
    """Choose the best available encoder for scientific false-colour output."""
    for name in ("h264_nvenc", "libx264rgb", "libx264", "ffv1", "png", "mpeg4"):
        if name in encoders:
            return name
    raise RuntimeError(
        "No supported ffmpeg video encoder found. "
        f"Available encoders: {', '.join(sorted(encoders)) or '<none>'}"
    )


def _encoder_config(chosen_encoder: str) -> tuple[list[str], str]:
    """Return ffmpeg flags and output filename for *chosen_encoder*."""
    if chosen_encoder == "h264_nvenc":
        return ([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-b:v", "6M",
            "-maxrate", "12M",
            "-pix_fmt", "yuv420p",
        ], "dynamics.mp4")
    if chosen_encoder == "libx264rgb":
        return ([
            "-c:v", "libx264rgb",
            "-preset", "fast",
            "-crf", "0",
            "-pix_fmt", "rgb24",
        ], "dynamics.mp4")
    if chosen_encoder == "libx264":
        return ([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "12",
            "-pix_fmt", "yuv444p",
        ], "dynamics.mp4")
    if chosen_encoder == "ffv1":
        return ([
            "-c:v", "ffv1",
            "-level", "3",
            "-pix_fmt", "rgb24",
        ], "dynamics.mkv")
    if chosen_encoder == "png":
        return ([
            "-c:v", "png",
            "-pix_fmt", "rgb24",
        ], "dynamics.mov")
    if chosen_encoder == "mpeg4":
        return ([
            "-c:v", "mpeg4",
            "-q:v", "1",
            "-bf", "0",
            "-pix_fmt", "yuv420p",
        ], "dynamics.mp4")
    return (["-c:v", chosen_encoder, "-pix_fmt", "yuv420p"], "dynamics.mp4")


def _read_field(h5: h5py.File, spec: dict[str, Any], idx: int) -> np.ndarray:
    raw = np.array(h5[f"fields/{spec['source']}"][idx])
    if spec.get("transform") == "abs2":
        return (np.abs(raw) ** 2).astype(np.float64)
    if np.iscomplexobj(raw):
        return raw.real.astype(np.float64)
    return raw.astype(np.float64)


def _to_uint8_rgb(
    arr: np.ndarray,
    lut: np.ndarray,
    vmin: float,
    vmax: float,
    norm_type: str | None,
    norm_gamma: float,
) -> np.ndarray:
    span = max(vmax - vmin, 1e-30)
    if norm_type == "power":
        indices = np.clip(
            ((np.clip(arr, vmin, vmax) - vmin) / span) ** norm_gamma * 255,
            0, 255,
        ).astype(np.uint8)
    else:
        indices = np.clip((arr - vmin) / span * 255, 0, 255).astype(np.uint8)
    return lut[indices]


def _format_scale_value(value: float) -> str:
    """Return a compact numeric label for colour-bar endpoints."""
    if not np.isfinite(value):
        return "nan"
    mag = abs(value)
    if mag != 0.0 and (mag < 1e-3 or mag >= 1e4):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _load_fonts(panel_width: int) -> tuple[object, object, object]:
    """Load fonts used by the fixed animation frame layout."""
    title_size = max(13, panel_width // 42)
    label_size = max(10, panel_width // 58)
    small_size = max(9, panel_width // 70)
    candidates = (
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    regular_candidates = (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    try:
        title_font = _PILFont.truetype(candidates[0], title_size)
    except (IOError, OSError):
        try:
            title_font = _PILFont.truetype(candidates[1], title_size)
        except (IOError, OSError):
            title_font = _PILFont.load_default()
    try:
        label_font = _PILFont.truetype(regular_candidates[0], label_size)
        small_font = _PILFont.truetype(regular_candidates[0], small_size)
    except (IOError, OSError):
        try:
            label_font = _PILFont.truetype(regular_candidates[1], label_size)
            small_font = _PILFont.truetype(regular_candidates[1], small_size)
        except (IOError, OSError):
            label_font = title_font
            small_font = title_font
    return title_font, label_font, small_font


def _text_size(draw: object, text: str, font: object) -> tuple[int, int]:
    """Return text dimensions for PIL versions with or without textbbox."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _draw_text_shadowed(
    draw: object,
    xy: tuple[int, int],
    text: str,
    font: object,
    fill: tuple[int, int, int] = _TEXT,
    shadow_offset: int = 1,
) -> None:
    """Draw readable text on the dark fixed-layout canvas."""
    x, y = xy
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=_SHADOW)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_rect(
    draw: object,
    xy: list[int],
    outline: tuple[int, int, int],
    width: int,
) -> None:
    """Draw a rectangle with stable thickness across Pillow versions."""
    x0, y0, x1, y1 = xy
    for off in range(max(1, width)):
        draw.rectangle([x0 - off, y0 - off, x1 + off, y1 + off], outline=outline)


def _draw_colorbar(
    img: object,
    draw: object,
    lut: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    vmin: float,
    vmax: float,
    font: object,
    shadow_offset: int,
) -> None:
    """Draw a compact horizontal fixed colour scale below one panel."""
    grad = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    bar = np.repeat(lut[grad], height, axis=0)
    img.paste(_PILImage.fromarray(bar, mode="RGB"), (x, y))
    _draw_rect(draw, [x, y, x + width, y + height], _PANEL_OUTLINE, 1)

    vmax_text = _format_scale_value(vmax)
    vmax_w, _ = _text_size(draw, vmax_text, font)
    label_y = y + height + max(2, height // 3)
    _draw_text_shadowed(
        draw,
        (x, label_y),
        _format_scale_value(vmin),
        font,
        fill=_TEXT,
        shadow_offset=shadow_offset,
    )
    _draw_text_shadowed(
        draw,
        (x + width - vmax_w, label_y),
        vmax_text,
        font,
        fill=_TEXT,
        shadow_offset=shadow_offset,
    )


def _draw_overlays(
    frame: np.ndarray,
    panel_width: int,
    panel_labels: list[str],
    timestamp_ps: float,
    lx_um: float,
    vmins: dict[str, float],
    vmaxs: dict[str, float],
    field_keys: list[str],
    luts: dict[str, np.ndarray],
) -> np.ndarray:
    """Compose a fixed-layout movie frame with panel borders and colour bars."""
    if not _PIL_AVAILABLE:
        raise RuntimeError("Pillow is required to render fixed-layout animation scales")

    n_panels = len(field_keys)
    if n_panels == 0:
        return frame

    panel_height = frame.shape[0]
    expected_width = n_panels * panel_width
    if frame.shape[1] < expected_width:
        raise ValueError(
            f"concatenated frame width {frame.shape[1]} is smaller than "
            f"{n_panels} panels × {panel_width}px"
        )

    title_font, label_font, small_font = _load_fonts(panel_width)
    margin = max(10, panel_width // 64)
    gap = max(14, panel_width // 48)
    top_band = max(46, panel_width // 20)
    colorbar_gap = max(8, panel_width // 96)
    colorbar_height = max(8, panel_width // 96)
    colorbar_label_height = max(14, panel_width // 50)
    scale_bar_band = max(28, panel_width // 34)
    block_width = panel_width
    canvas_width = 2 * margin + n_panels * block_width + (n_panels - 1) * gap
    canvas_height = (
        2 * margin
        + top_band
        + panel_height
        + colorbar_gap
        + colorbar_height
        + colorbar_label_height
        + scale_bar_band
    )

    img = _PILImage.new("RGB", (canvas_width, canvas_height), _FRAME_BG)
    draw = _PILDraw.Draw(img)
    shadow_offset = max(1, panel_width // 200)
    data_y = margin + top_band

    ts_text = f"t = {timestamp_ps:.1f} ps"
    ts_w, _ = _text_size(draw, ts_text, title_font)
    _draw_text_shadowed(
        draw,
        ((canvas_width - ts_w) // 2, margin),
        ts_text,
        title_font,
        shadow_offset=shadow_offset,
    )

    for i, label in enumerate(panel_labels):
        key = field_keys[i]
        block_x = margin + i * (block_width + gap)
        crop = frame[:, i * panel_width:(i + 1) * panel_width]
        panel_img = _PILImage.fromarray(crop, mode="RGB")
        img.paste(panel_img, (block_x, data_y))

        _draw_rect(
            draw,
            [block_x, data_y, block_x + panel_width, data_y + panel_height],
            _PANEL_OUTLINE,
            max(1, panel_width // 256),
        )
        _draw_rect(
            draw,
            [
                block_x - 3,
                data_y - 3,
                block_x + panel_width + 3,
                data_y + panel_height + colorbar_gap + colorbar_height + colorbar_label_height,
            ],
            _PANEL_SUBTLE,
            1,
        )

        _draw_text_shadowed(
            draw,
            (block_x, margin + max(20, panel_width // 42)),
            label,
            label_font,
            shadow_offset=shadow_offset,
        )

        colorbar_y = data_y + panel_height + colorbar_gap
        _draw_colorbar(
            img,
            draw,
            luts[key],
            block_x,
            colorbar_y,
            panel_width,
            colorbar_height,
            vmins.get(key, 0.0),
            vmaxs.get(key, 1.0),
            small_font,
            shadow_offset,
        )

    bar_px = int(round(_SCALEBAR_LENGTH_UM / max(lx_um, 1e-12) * panel_width))
    bar_px = max(bar_px, 4)
    bar_px = min(bar_px, max(4, panel_width - 2 * margin))
    bar_thickness = max(2, panel_width // 150)
    bar_x0 = margin
    bar_y0 = (
        data_y
        + panel_height
        + colorbar_gap
        + colorbar_height
        + colorbar_label_height
        + max(7, scale_bar_band // 4)
    )
    bar_x1 = bar_x0 + bar_px
    draw.rectangle(
        [
            bar_x0 - shadow_offset,
            bar_y0 + shadow_offset,
            bar_x1 + shadow_offset,
            bar_y0 + bar_thickness + shadow_offset,
        ],
        fill=_SHADOW,
    )
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y0 + bar_thickness], fill=_TEXT)
    bar_label = f"{_SCALEBAR_LENGTH_UM:.0f} µm"
    _draw_text_shadowed(
        draw,
        (bar_x0, bar_y0 + bar_thickness + 2),
        bar_label,
        small_font,
        shadow_offset=shadow_offset,
    )

    return np.array(img)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    ph = h + (h % 2)
    pw = w + (w % 2)
    if ph == h and pw == w:
        return frame
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:h, :w] = frame
    return padded


def _coerce_clim(raw_clim: object) -> tuple[float, float] | None:
    """Return a validated ``(vmin, vmax)`` tuple or ``None``."""
    if raw_clim is None:
        return None
    if not isinstance(raw_clim, (tuple, list)) or len(raw_clim) != 2:
        raise ValueError(f"field spec clim must be a 2-element tuple/list, got {raw_clim!r}")
    vmin, vmax = float(raw_clim[0]), float(raw_clim[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"field spec clim must be finite with vmax > vmin, got {raw_clim!r}")
    return vmin, vmax


def _default_zero_vmin(spec: dict[str, Any]) -> bool:
    """Return whether a field should use physical zero as black by default."""
    if "zero_vmin" in spec:
        return bool(spec["zero_vmin"])
    if spec.get("transform") == "abs2":
        return True
    return str(spec.get("source", "")) in {"Pump", "nA", "nI", "nR"}


def _global_vmax_attr_name(key: str, spec: dict[str, Any]) -> str | None:
    """Return the HDF5 attribute carrying an exact full-run max for this field."""
    source = str(spec.get("source", key))
    if source == "psi" and spec.get("transform") == "abs2":
        return "animation_global_max_psi_sq"
    if source in {"nA", "nI", "Pump"}:
        return f"animation_global_max_{source}"
    return None


def _read_global_vmax_hint(h5: h5py.File, key: str, spec: dict[str, Any]) -> float | None:
    """Read a full-run maximum written by the simulation loop, if present."""
    attr = _global_vmax_attr_name(key, spec)
    if attr is None or attr not in h5.attrs:
        return None
    value = float(h5.attrs[attr])
    if not np.isfinite(value):
        return None
    return value


def _scan_global_color_limits(
    h5: h5py.File,
    field_specs: dict[str, dict],
    physical_indices: list[int],
) -> tuple[dict[str, float], dict[str, float]]:
    """Scan all rendered frames and return global colour limits per field."""
    vmins: dict[str, float] = {}
    vmaxs: dict[str, float] = {}

    for key, spec in field_specs.items():
        fixed_clim = _coerce_clim(spec.get("clim"))
        if fixed_clim is not None:
            vmins[key], vmaxs[key] = fixed_clim
            continue

        local_min = 0.0 if _default_zero_vmin(spec) else float("inf")
        local_max = float("-inf")
        for physical_idx in physical_indices:
            arr = _read_field(h5, spec, physical_idx)
            if not _default_zero_vmin(spec):
                local_min = min(local_min, float(np.nanmin(arr)))
            local_max = max(local_max, float(np.nanmax(arr)))

        vmax_hint = _read_global_vmax_hint(h5, key, spec)
        if vmax_hint is not None:
            local_max = max(local_max, vmax_hint)

        if not np.isfinite(local_min):
            local_min = 0.0
        if not np.isfinite(local_max):
            local_max = local_min + 1e-12
        if local_max <= local_min:
            local_max = local_min + 1e-12
        vmins[key] = local_min
        vmaxs[key] = local_max

    return vmins, vmaxs


def _downscale_field(arr: np.ndarray, factor: int) -> np.ndarray:
    """Area-average a 2D field by an integer factor."""
    if factor <= 1:
        return arr
    h, w = arr.shape[:2]
    nh = h // factor * factor
    nw = w // factor * factor
    cropped = arr[:nh, :nw]
    return cropped.reshape(nh // factor, factor, nw // factor, factor).mean(axis=(1, 3))


def generate_animation(
    routine: str,
    field_specs: dict[str, dict],
    extent: list[float],
    data_dir: str,
    results_dir: str,
    fps: int = ANIM_FPS,
    encoder: str | None = None,
    downscale_factor: int = 1,
) -> None:
    """Stream an animation from scratch-local HDF5 to a video via ffmpeg.

    Parameters
    ----------
    routine : str
        Scenario name; HDF5 read from ``{data_dir}/{routine}.h5``.
    field_specs : dict
        Mapping from display key to spec dict with keys: ``source``,
        ``cmap``, optionally ``transform``, ``norm``, ``norm_gamma``,
        ``zero_vmin`` and fixed ``clim``.
    extent : list[float]
        ``[xmin, xmax, ymin, ymax]`` µm — used for scale bar calculation.
    data_dir : str
        Directory containing ``{routine}.h5``.
    results_dir : str
        Output base directory; file written to ``{results_dir}/{routine}/dynamics.*``.
    fps : int
        Output frame rate.
    encoder : str or None
        Override the video encoder. Falls back to ``RENDER_ENCODER`` env var,
        then an automatic best-available encoder choice.
    downscale_factor : int
        Spatially downsample each field panel by this integer factor
        (area-average) before colormapping. Reduces video resolution
        without affecting simulation fidelity.
    """
    ffmpeg = _resolve_ffmpeg()
    encoders = _available_encoders(ffmpeg)
    chosen_encoder = encoder or os.environ.get("RENDER_ENCODER") or _pick_default_encoder(encoders)
    if chosen_encoder not in encoders:
        raise RuntimeError(
            f"Requested ffmpeg encoder '{chosen_encoder}' is not available in '{ffmpeg}'.\n"
            f"Available encoders: {', '.join(sorted(encoders)) or '<none>'}"
        )

    h5_path = os.path.join(data_dir, f"{routine}.h5")
    out_dir = os.path.join(results_dir, routine)
    os.makedirs(out_dir, exist_ok=True)
    encoder_flags, out_name = _encoder_config(chosen_encoder)
    out_path = os.path.join(out_dir, out_name)

    field_keys = list(field_specs.keys())
    luts = {k: _get_lut(field_specs[k]["cmap"]) for k in field_keys}
    panel_labels = [_PANEL_DISPLAY_NAMES.get(k, k) for k in field_keys]
    lx_um = float(extent[1] - extent[0]) if extent and len(extent) >= 2 else 64.0

    with h5py.File(h5_path, "r") as h5:
        sort_order = np.argsort(h5["time"][:], kind="stable")
        times = h5["time"][:]
        n_total = len(sort_order)

        target_frames = fps * ANIM_TARGET_SECONDS
        stride = max(1, n_total // target_frames)
        anim_logical = list(range(0, n_total, stride))
        anim_physical = [int(sort_order[i]) for i in anim_logical]
        anim_times = [float(times[sort_order[i]]) for i in anim_logical]

        full_physical = [int(i) for i in sort_order]
        print(
            f"    Scanning global colour limits from {len(full_physical)} "
            "recorded frames ..."
        )
        vmins, vmaxs = _scan_global_color_limits(h5, field_specs, full_physical)
        for k in field_keys:
            print(f"      {k}: vmin={vmins[k]:.6g}, vmax={vmaxs[k]:.6g}")

        probe_panels = [
            _to_uint8_rgb(
                _downscale_field(_read_field(h5, field_specs[k], anim_physical[0]), downscale_factor),
                luts[k], vmins[k], vmaxs[k], field_specs[k].get("norm"),
                float(field_specs[k].get("norm_gamma", PUMP_NORM_GAMMA)),
            )
            for k in field_keys
        ]
        probe_raw = np.concatenate(probe_panels, axis=1)
        panel_width = probe_panels[0].shape[1]
        probe_frame = _pad_even(_draw_overlays(
            probe_raw, panel_width, panel_labels, anim_times[0],
            lx_um, vmins, vmaxs, field_keys, luts,
        ))
        height, width = probe_frame.shape[:2]

        n_frames = len(anim_physical)
        print(
            f"    Streaming {n_frames} frames  {height}×{width}px  "
            f"encoder={chosen_encoder}  fps={fps} ..."
        )

        cmd = [
            ffmpeg, "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            *encoder_flags,
            out_path,
        ]

        # Never use subprocess.PIPE for stderr here. ffmpeg can write enough
        # diagnostics/progress output to fill the pipe while we are writing raw
        # frames to stdin, which deadlocks long movie renders. A temporary file
        # keeps error diagnostics without blocking the streaming loop.
        with tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_file)
            try:
                for physical_idx, t_ps in zip(anim_physical, anim_times):
                    panels = [
                        _to_uint8_rgb(
                            _downscale_field(_read_field(h5, field_specs[k], physical_idx), downscale_factor),
                            luts[k], vmins[k], vmaxs[k], field_specs[k].get("norm"),
                            float(field_specs[k].get("norm_gamma", PUMP_NORM_GAMMA)),
                        )
                        for k in field_keys
                    ]
                    raw = np.concatenate(panels, axis=1)
                    annotated = _draw_overlays(
                        raw, panel_width, panel_labels, t_ps,
                        lx_um, vmins, vmaxs, field_keys, luts,
                    )
                    proc.stdin.write(_pad_even(annotated).tobytes())
            except BrokenPipeError:
                proc.stdin.close()
                proc.wait()
                stderr_file.seek(0, os.SEEK_END)
                size = stderr_file.tell()
                stderr_file.seek(max(0, size - 4096))
                stderr_bytes = stderr_file.read()
                raise RuntimeError(
                    f"ffmpeg terminated unexpectedly during streaming to {out_path}.\n"
                    f"ffmpeg stderr: {stderr_bytes.decode(errors='replace')[-1000:]}"
                )

            proc.stdin.close()
            proc.wait()
            stderr_file.seek(0, os.SEEK_END)
            size = stderr_file.tell()
            stderr_file.seek(max(0, size - 4096))
            stderr_bytes = stderr_file.read()
            ret = proc.returncode
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {ret}.  Output: {out_path}\n"
                f"ffmpeg stderr: {stderr_bytes.decode(errors='replace')[-1000:]}"
            )

    print(f"    {out_path}")
