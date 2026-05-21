"""Streaming animation renderer.

Renders scenario fields from an HDF5 file to a video file via ffmpeg.
Reads one frame at a time — no whole-animation preload into RAM.

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


ANIM_FPS = 8
ANIM_TARGET_SECONDS = 60
PUMP_NORM_GAMMA = 0.3
N_SAMPLE_FRAMES = 20


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
) -> np.ndarray:
    span = max(vmax - vmin, 1e-30)
    if norm_type == "power":
        indices = np.clip(
            ((np.clip(arr, vmin, vmax) - vmin) / span) ** PUMP_NORM_GAMMA * 255,
            0, 255,
        ).astype(np.uint8)
    else:
        indices = np.clip((arr - vmin) / span * 255, 0, 255).astype(np.uint8)
    return lut[indices]


def _pad_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    ph = h + (h % 2)
    pw = w + (w % 2)
    if ph == h and pw == w:
        return frame
    padded = np.zeros((ph, pw, 3), dtype=np.uint8)
    padded[:h, :w] = frame
    return padded


def generate_animation(
    routine: str,
    field_specs: dict[str, dict],
    extent: list[float],
    data_dir: str,
    results_dir: str,
    fps: int = ANIM_FPS,
    encoder: str | None = None,
) -> None:
    """Stream an animation from scratch-local HDF5 to a video via ffmpeg.

    Parameters
    ----------
    routine : str
        Scenario name; HDF5 read from ``{data_dir}/{routine}.h5``.
    field_specs : dict
        Mapping from display key to spec dict with keys: ``source``,
        ``cmap``, optionally ``transform`` and ``norm``.
    extent : list[float]
        ``[xmin, xmax, ymin, ymax]`` µm — kept for API symmetry, unused here.
    data_dir : str
        Directory containing ``{routine}.h5``.
    results_dir : str
        Output base directory; file written to ``{results_dir}/{routine}/dynamics.*``.
    fps : int
        Output frame rate.
    encoder : str or None
        Override the video encoder. Falls back to ``RENDER_ENCODER`` env var,
        then an automatic best-available encoder choice.
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

    with h5py.File(h5_path, "r") as h5:
        sort_order = np.argsort(h5["time"][:], kind="stable")
        n_total = len(sort_order)

        target_frames = fps * ANIM_TARGET_SECONDS
        stride = max(1, n_total // target_frames)
        anim_logical = list(range(0, n_total, stride))
        anim_physical = [int(sort_order[i]) for i in anim_logical]

        sample_stride = max(1, (n_total - 1) // max(1, N_SAMPLE_FRAMES - 1))
        sample_physical = [
            int(sort_order[i]) for i in range(0, n_total, sample_stride)
        ][:N_SAMPLE_FRAMES]

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
        probe_frame = _pad_even(np.concatenate(probe_panels, axis=1))
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
                for physical_idx in anim_physical:
                    panels = [
                        _to_uint8_rgb(
                            _read_field(h5, field_specs[k], physical_idx),
                            luts[k], vmins[k], vmaxs[k], field_specs[k].get("norm"),
                        )
                        for k in field_keys
                    ]
                    proc.stdin.write(_pad_even(np.concatenate(panels, axis=1)).tobytes())
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
