"""Video encoder abstraction for online animation streaming.

Backend selection priority:
  1. PyNvVideoCodec  (if importable and ``backend="auto"`` or ``"pynvvideocodec"``)
  2. FFmpeg subprocess with the requested NVENC/CPU encoder
  3. CPU libx264 fallback with a logged warning

stderr is never read via subprocess.PIPE during streaming to avoid deadlock;
a TemporaryFile is used instead and its tail is surfaced on encoder failure.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ENCODER_FLAGS: dict[str, list[str]] = {
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-b:v", "6M", "-maxrate", "12M", "-pix_fmt", "yuv420p"],
    "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p4", "-rc", "vbr",
                   "-b:v", "8M", "-maxrate", "16M", "-pix_fmt", "yuv420p"],
    "libx264rgb": ["-c:v", "libx264rgb", "-preset", "fast", "-crf", "0",
                   "-pix_fmt", "rgb24"],
    "libx264":    ["-c:v", "libx264", "-preset", "fast", "-crf", "12",
                   "-pix_fmt", "yuv444p"],
    "ffv1":       ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"],
    "mpeg4":      ["-c:v", "mpeg4", "-q:v", "1", "-bf", "0", "-pix_fmt", "yuv420p"],
}

_ENCODER_EXTENSION: dict[str, str] = {
    "h264_nvenc": ".mp4",
    "hevc_nvenc": ".mp4",
    "libx264rgb": ".mp4",
    "libx264":    ".mp4",
    "ffv1":       ".mkv",
    "mpeg4":      ".mp4",
}

_CPU_FALLBACK_ORDER = ["libx264rgb", "libx264", "ffv1", "mpeg4"]


class VideoEncoder(ABC):
    """Abstract video encoder interface."""

    @abstractmethod
    def open(self, width: int, height: int, fps: int, output_path: str) -> None:
        """Open the encoder for writing frames of the given shape."""

    @abstractmethod
    def write_frame(self, frame: Any) -> None:
        """Write one uint8 RGB frame (NumPy or CuPy array, shape H×W×3)."""

    @abstractmethod
    def close(self) -> None:
        """Flush, finalize, and close the encoder."""

    def abort(self) -> None:
        """Force-close the encoder without checking exit status.

        Called after a streaming failure to ensure the subprocess is
        terminated.  Implementations must not raise.
        """


class FFmpegRawVideoEncoder(VideoEncoder):
    """FFmpeg subprocess encoder fed via rawvideo pipe.

    Pinned host memory is allocated when CuPy is active so that D2H copies
    use DMA rather than pageable transfer.  ``stderr`` is routed to a
    TemporaryFile to prevent pipe-buffer deadlocks during long writes.
    """

    def __init__(self, encoder_name: str, ffmpeg_bin: str) -> None:
        self._encoder_name = encoder_name
        self._ffmpeg = ffmpeg_bin
        self._proc: subprocess.Popen | None = None
        self._stderr_file: Any = None
        self._pinned_buf: np.ndarray | None = None
        self._pinned_registered = False
        self._output_path = ""

    def open(self, width: int, height: int, fps: int, output_path: str) -> None:
        self._output_path = output_path
        flags = _ENCODER_FLAGS.get(
            self._encoder_name,
            ["-c:v", self._encoder_name, "-pix_fmt", "yuv420p"],
        )
        cmd = [
            self._ffmpeg, "-y",
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            *flags,
            output_path,
        ]
        self._stderr_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._stderr_file)

        self._pinned_buf = np.empty((height, width, 3), dtype=np.uint8)
        try:
            import cupy as cp
            cp.cuda.runtime.hostRegister(
                self._pinned_buf.ctypes.data, self._pinned_buf.nbytes, 0
            )
            self._pinned_registered = True
        except Exception:
            self._pinned_registered = False

    def write_frame(self, frame: Any) -> None:
        if self._proc is None:
            raise RuntimeError("Encoder not opened")
        if hasattr(frame, "get"):
            if self._pinned_buf is not None:
                frame.get(out=self._pinned_buf)
                buf = self._pinned_buf
            else:
                buf = frame.get()
        else:
            buf = np.asarray(frame)
        try:
            self._proc.stdin.write(memoryview(buf))
        except BrokenPipeError:
            self._proc.stdin.close()
            self._proc.wait()
            tail = self._stderr_tail()
            raise RuntimeError(
                f"ffmpeg terminated unexpectedly during streaming to {self._output_path}.\n"
                f"stderr: {tail}"
            )

    def close(self) -> None:
        if self._proc is None:
            return
        if self._pinned_registered and self._pinned_buf is not None:
            try:
                import cupy as cp
                cp.cuda.runtime.hostUnregister(self._pinned_buf.ctypes.data)
            except Exception:
                pass
        self._proc.stdin.close()
        self._proc.wait()
        tail = self._stderr_tail()
        ret = self._proc.returncode
        if self._stderr_file is not None:
            self._stderr_file.close()
        self._proc = None
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {ret} for {self._output_path}.\n"
                f"stderr: {tail}"
            )

    def abort(self) -> None:
        if self._proc is None:
            return
        if self._pinned_registered and self._pinned_buf is not None:
            try:
                import cupy as cp
                cp.cuda.runtime.hostUnregister(self._pinned_buf.ctypes.data)
            except Exception:
                pass
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
        self._proc = None

    def _stderr_tail(self, max_bytes: int = 4096) -> str:
        if self._stderr_file is None:
            return ""
        try:
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            self._stderr_file.seek(max(0, size - max_bytes))
            return self._stderr_file.read().decode(errors="replace")
        except Exception:
            return ""


def _resolve_ffmpeg() -> str:
    explicit = os.environ.get("FFMPEG_BIN")
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg binary not found. Set FFMPEG_BIN to its path."
    )


def _available_encoders(ffmpeg_bin: str) -> set[str]:
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


def output_extension_for(encoder_name: str) -> str:
    """Return the recommended file extension for *encoder_name*."""
    return _ENCODER_EXTENSION.get(encoder_name, ".mp4")


def create_encoder(
    backend: str = "auto",
    encoder_name: str = "h264_nvenc",
    ffmpeg_bin: str | None = None,
) -> tuple[VideoEncoder, str]:
    """Select and instantiate the best available video encoder.

    Parameters
    ----------
    backend : str
        ``"auto"``, ``"pynvvideocodec"``, or ``"ffmpeg"``.
    encoder_name : str
        Preferred encoder for the ffmpeg backend (e.g. ``"h264_nvenc"``).
    ffmpeg_bin : str or None
        Explicit path to ffmpeg; resolved via ``FFMPEG_BIN`` env var or
        ``PATH`` when ``None``.

    Returns
    -------
    tuple[VideoEncoder, str]
        The encoder instance and the resolved encoder name (which may differ
        from the requested name when a fallback was selected).
    """
    if backend == "pynvvideocodec":
        raise RuntimeError(
            "Animation backend 'pynvvideocodec' is not yet implemented. "
            "Use backend='auto' or backend='ffmpeg'."
        )
    if backend == "auto":
        pass

    resolved_ffmpeg = ffmpeg_bin or _resolve_ffmpeg()
    available = _available_encoders(resolved_ffmpeg)

    if encoder_name in available:
        chosen = encoder_name
    else:
        chosen = next((e for e in _CPU_FALLBACK_ORDER if e in available), None)
        if chosen is None:
            raise RuntimeError(
                f"No supported ffmpeg encoder found. "
                f"Requested '{encoder_name}'. "
                f"Available: {', '.join(sorted(available)) or '<none>'}"
            )
        logger.warning(
            "Encoder '%s' not available in ffmpeg; falling back to '%s'.",
            encoder_name, chosen,
        )
        import warnings
        warnings.warn(
            f"Animation encoder '{encoder_name}' not available; "
            f"falling back to '{chosen}' (CPU encoding, slower).",
            RuntimeWarning,
            stacklevel=2,
        )

    return FFmpegRawVideoEncoder(chosen, resolved_ffmpeg), chosen


def preflight_encode(encoder_name: str, ffmpeg_bin: str | None = None) -> None:
    """Verify the encoder can actually produce output on this node.

    Encodes a minimal valid stream (8 frames at 128×128) and deletes the
    temporary file.  Raises :exc:`RuntimeError` on any failure so that
    the caller can abort *before* a long simulation begins.

    Parameters
    ----------
    encoder_name : str
        The encoder to validate (e.g. ``"h264_nvenc"``).
    ffmpeg_bin : str or None
        Explicit path to ffmpeg; resolved automatically when ``None``.
    """
    import tempfile
    import numpy as np

    resolved = ffmpeg_bin or _resolve_ffmpeg()
    width, height, fps, n_frames = 128, 128, 4, 8
    black = np.zeros((height, width, 3), dtype=np.uint8)

    with tempfile.NamedTemporaryFile(suffix=output_extension_for(encoder_name), delete=False) as f:
        tmp_path = f.name

    try:
        enc = FFmpegRawVideoEncoder(encoder_name, resolved)
        enc.open(width, height, fps, tmp_path)
        for _ in range(n_frames):
            enc.write_frame(black)
        enc.close()
    except Exception as exc:
        raise RuntimeError(
            f"Encode preflight failed for encoder '{encoder_name}' "
            f"using '{resolved}': {exc}\n"
            "Check that the GPU driver is available and the encoder is supported "
            "on this node.  Set FFMPEG_BIN or RENDER_ENCODER to override."
        ) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
