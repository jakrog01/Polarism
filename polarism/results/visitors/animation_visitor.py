"""Online GPU-side animation visitor for polarism simulations.

Two usage modes
---------------
**ResultVisitor mode** (via ResultsManager)
    Attach to a :class:`~polarism.results.results_manager.ResultsManager`.
    ``panel_specs[i].source`` must match the name of a registered
    :class:`~polarism.results.result_node.ResultNode`.  ``transform`` should
    be ``None`` because node ``compute_fn`` already applies the transform.
    ``data_location = "device"`` causes the manager to deliver backend-native
    arrays.

**Direct mode** (pump_multi_comparison pipeline)
    Call :meth:`record_frame` with a raw-fields dict.  ``panel_specs[i].source``
    is the key in that dict and ``transform`` is applied by this visitor.

Colour-range sampling
---------------------
Auto colour limits are derived from the first ``sample_frames`` frames.
For multi-pulse experiments where the condensate only appears after many
pulses this will saturate.  Set ``clim`` in each :class:`AnimationFieldSpec`
to override and avoid saturation.  A ``clim_margin`` fraction (default 10 %)
expands the sampled range as a safety buffer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polarism.compute_engine import compute_engine
from polarism.results.rendering.gpu_movie import (
    apply_transform,
    build_lut,
    render_panels,
)
from polarism.results.rendering.video_encoder import (
    VideoEncoder,
    create_encoder,
    output_extension_for,
)
from polarism.results.visitors.result_visitor import ResultVisitor

logger = logging.getLogger(__name__)


@dataclass
class AnimationFieldSpec:
    """Describe one false-colour panel in the rendered animation.

    Parameters
    ----------
    source : str
        Key in the raw-fields dict (direct mode) *or* the node name (visitor
        mode).  In visitor mode ``transform`` should normally be ``None``.
    cmap : str
        Matplotlib colormap name.
    transform : str or None
        ``"abs2"`` → |raw|², ``"kspace_log"`` → log₁₀ normalised k-space
        power, ``None`` → pass-through / real part.
    norm : str or None
        ``"power"`` for gamma-compressed mapping, ``None`` for linear.
    norm_gamma : float
        Exponent used when ``norm="power"``.
    clim : tuple of float or None
        Fixed ``(vmin, vmax)`` colour limits.  ``None`` → auto-sample from
        the first ``sample_frames`` frames.  **Set this for scientific
        accuracy when the field range changes significantly across the run.**
    """
    source: str
    cmap: str
    transform: str | None = None
    norm: str | None = None
    norm_gamma: float = 0.3
    clim: tuple[float, float] | None = None


class AnimationVisitor(ResultVisitor):
    """Stream simulation field frames to a video file during the simulation loop.

    Inherits from :class:`~polarism.results.visitors.result_visitor.ResultVisitor`
    so it can be attached to :class:`~polarism.results.results_manager.ResultsManager`
    and receive backend-native arrays without forcing a CPU conversion.

    Parameters
    ----------
    output_path : str
        Video output path.  The file extension may be replaced based on the
        selected encoder (e.g. ``.mp4`` → ``.mkv`` for ``ffv1``).
    panel_specs : list of AnimationFieldSpec
        Panels to render, left-to-right.
    fps : int
        Output frame rate.
    target_seconds : int
        Informational; used for log messages.
    backend : str
        ``"auto"``, ``"pynvvideocodec"``, or ``"ffmpeg"``.
    encoder_name : str
        Preferred ffmpeg encoder (e.g. ``"h264_nvenc"``).
    sample_frames : int
        Number of frames to buffer before computing auto colour limits.
        Ignored for panels with a fixed ``clim``.
    clim_margin : float
        Fraction by which to expand the sampled colour range on each side.
        Provides a safety buffer against per-frame maxima that slightly
        exceed the sampled maximum.
    log_every : int
        Print a progress line every this many encoded frames.
    """

    data_location: str = "device"
    fatal_on_error: bool = False

    def __init__(
        self,
        output_path: str,
        panel_specs: list[AnimationFieldSpec],
        fps: int = 8,
        target_seconds: int = 60,
        backend: str = "auto",
        encoder_name: str = "h264_nvenc",
        sample_frames: int = 50,
        clim_margin: float = 0.10,
        log_every: int = 50,
    ) -> None:
        self._output_path = output_path
        self._panel_specs = panel_specs
        self._fps = fps
        self._target_seconds = target_seconds
        self._backend = backend
        self._encoder_name = encoder_name
        self._sample_frames = sample_frames
        self._clim_margin = clim_margin
        self._log_every = log_every

        self._xp = compute_engine.xp
        self._luts: list[Any] = [build_lut(self._xp, spec.cmap) for spec in panel_specs]

        self._sample_buf: list[list[Any]] = []
        self._vmins: list[float] | None = None
        self._vmaxs: list[float] | None = None

        self._encoder: VideoEncoder | None = None
        self._resolved_encoder_name: str = encoder_name
        self._total_frames: int = 0
        self._t0_open: float | None = None

        self._t_render_total: float = 0.0
        self._t_write_total: float = 0.0

        self.status: str = "running"
        self.error_message: str | None = None

        logger.info(
            "AnimationVisitor: %d panels, fps=%d, backend=%s, encoder=%s, output=%s",
            len(panel_specs), fps, backend, encoder_name, output_path,
        )
        print(
            f"  [animation] {len(panel_specs)} panels  fps={fps}  "
            f"backend={backend}  encoder={encoder_name}",
            flush=True,
        )

    def needs(self, nodes: list) -> set[str]:
        """Return node names this visitor requires when used via ResultsManager."""
        return {spec.source for spec in self._panel_specs}

    def visit_all(self, nodes: list, t: float = 0.0, cached: dict | None = None, **context) -> None:
        """Receive pre-transformed device arrays from ResultsManager.

        ``cached`` maps node name → ``(field_device, reduced_device)``.
        ``transform`` in each ``AnimationFieldSpec`` is **not** re-applied
        because the node's ``compute_fn`` already transformed the field.
        """
        if cached is None:
            return
        needed = self.needs(nodes)
        if not all(spec.source in cached for spec in self._panel_specs if spec.source in needed):
            return
        already_transformed = [cached[spec.source][0] for spec in self._panel_specs]
        self._encode_frame(already_transformed, t=t)

    def record_frame(self, raw_fields: dict[str, Any], t: float | None = None) -> None:
        """Accept one raw-field frame and apply transforms (direct-call mode).

        Parameters
        ----------
        raw_fields : dict
            Mapping from field key to backend array (CuPy or NumPy).
        t : float or None
            Current simulation time in ps, for progress logging.
        """
        t_r0 = time.monotonic()
        transformed = [
            apply_transform(self._xp, raw_fields[spec.source], spec.transform)
            for spec in self._panel_specs
        ]
        self._t_render_total += time.monotonic() - t_r0
        self._encode_frame(transformed, t=t)

    def _encode_frame(self, transformed: list[Any], t: float | None) -> None:
        """Core encode path shared by visit_all and record_frame."""
        if self._encoder is None:
            all_fixed = all(spec.clim is not None for spec in self._panel_specs)
            if all_fixed and not self._sample_buf:
                self._vmins = [spec.clim[0] for spec in self._panel_specs]
                self._vmaxs = [spec.clim[1] for spec in self._panel_specs]
                self._open_encoder(transformed)
                self._write_one(transformed)
                print("  [animation] first frame encoded", flush=True)
                return

            self._sample_buf.append([
                arr.astype(self._xp.float32) if hasattr(arr, "astype") else arr
                for arr in transformed
            ])

            if len(self._sample_buf) >= self._sample_frames:
                self._vmins, self._vmaxs = self._stats_from_buffer()
                self._open_encoder(self._sample_buf[0])
                first = True
                for frame_arrays in self._sample_buf:
                    self._write_one(frame_arrays)
                    if first:
                        print("  [animation] first frame encoded", flush=True)
                        first = False
                self._sample_buf.clear()
            return

        t_r0 = time.monotonic()
        frame = render_panels(
            self._xp, transformed,
            [s.norm for s in self._panel_specs],
            [s.norm_gamma for s in self._panel_specs],
            self._luts, self._vmins, self._vmaxs,
        )
        self._t_render_total += time.monotonic() - t_r0

        t_w0 = time.monotonic()
        self._encoder.write_frame(frame)
        self._t_write_total += time.monotonic() - t_w0
        self._total_frames += 1

        if self._total_frames % self._log_every == 0:
            elapsed = time.monotonic() - (self._t0_open or time.monotonic())
            fps_actual = self._total_frames / max(elapsed, 1e-6)
            t_str = f"  t={t:.1f} ps" if t is not None else ""
            print(
                f"  [animation] frame {self._total_frames}{t_str}  "
                f"{fps_actual:.1f} frames/s",
                flush=True,
            )

    def _stats_from_buffer(self) -> tuple[list[float], list[float]]:
        n = len(self._panel_specs)
        vmins: list[float] = [float("inf")] * n
        vmaxs: list[float] = [float("-inf")] * n
        for frame_arrays in self._sample_buf:
            for i, spec in enumerate(self._panel_specs):
                if spec.clim is not None:
                    vmins[i] = spec.clim[0]
                    vmaxs[i] = spec.clim[1]
                else:
                    vmins[i] = min(vmins[i], float(frame_arrays[i].min()))
                    vmaxs[i] = max(vmaxs[i], float(frame_arrays[i].max()))
        for i in range(n):
            if vmaxs[i] <= vmins[i]:
                vmaxs[i] = vmins[i] + 1e-12
            if self._panel_specs[i].clim is None and self._clim_margin > 0.0:
                span = vmaxs[i] - vmins[i]
                vmins[i] -= self._clim_margin * span
                vmaxs[i] += self._clim_margin * span
        return vmins, vmaxs

    def _open_encoder(self, probe_transformed: list[Any]) -> None:
        probe_frame = render_panels(
            self._xp, probe_transformed,
            [s.norm for s in self._panel_specs],
            [s.norm_gamma for s in self._panel_specs],
            self._luts, self._vmins, self._vmaxs,
        )
        probe_cpu = probe_frame.get() if hasattr(probe_frame, "get") else probe_frame
        import numpy as np
        probe_cpu = np.asarray(probe_cpu)
        height, width = probe_cpu.shape[:2]

        base = self._output_path
        if "." in base:
            base = base.rsplit(".", 1)[0]

        encoder, resolved = create_encoder(self._backend, self._encoder_name)
        self._resolved_encoder_name = resolved
        ext = output_extension_for(resolved)
        out_path = base + ext
        self._output_path = out_path

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        encoder.open(width, height, self._fps, out_path)
        self._encoder = encoder
        self._t0_open = time.monotonic()

        print(
            f"  [animation] encoder={resolved}  {width}×{height}  "
            f"fps={self._fps}  → {out_path}",
            flush=True,
        )
        logger.info(
            "Animation encoder opened: %s  %dx%d @ %d fps  %s",
            resolved, width, height, self._fps, out_path,
        )

    def _write_one(self, transformed: list[Any]) -> None:
        t_r0 = time.monotonic()
        frame = render_panels(
            self._xp, transformed,
            [s.norm for s in self._panel_specs],
            [s.norm_gamma for s in self._panel_specs],
            self._luts, self._vmins, self._vmaxs,
        )
        self._t_render_total += time.monotonic() - t_r0
        t_w0 = time.monotonic()
        self._encoder.write_frame(frame)
        self._t_write_total += time.monotonic() - t_w0
        self._total_frames += 1

    def abort(self, exc: BaseException | None = None) -> None:
        """Best-effort encoder cleanup after a streaming failure.

        Terminates the ffmpeg subprocess without checking exit status.
        Sets :attr:`status` to ``"failed"`` and records the error message.
        Safe to call from an ``except`` block; never raises.
        """
        if exc is not None:
            self.error_message = str(exc)
        self.status = "failed"
        if self._encoder is not None:
            try:
                self._encoder.abort()
            except Exception:
                pass
            self._encoder = None
        self._sample_buf.clear()

    def finalize(self) -> None:
        """Flush buffered frames, close encoder, and report timing.

        Safe to call multiple times; idempotent after the first call.
        """
        if self._encoder is None and self._sample_buf:
            if self._vmins is None:
                self._vmins, self._vmaxs = self._stats_from_buffer()
            if self._vmins is not None:
                self._open_encoder(self._sample_buf[0])
                first = True
                for frame_arrays in self._sample_buf:
                    self._write_one(frame_arrays)
                    if first:
                        print("  [animation] first frame encoded", flush=True)
                        first = False
                self._sample_buf.clear()

        if self._encoder is None:
            if self.status != "failed":
                print("  [animation] no frames recorded; nothing to finalize", flush=True)
            return

        t_close0 = time.monotonic()
        self._encoder.close()
        self._encoder = None
        t_close = time.monotonic() - t_close0
        self.status = "ok"

        total_elapsed = time.monotonic() - (self._t0_open or time.monotonic())
        print(
            f"  [animation] finalized  {self._total_frames} frames  "
            f"render={self._t_render_total:.2f}s  "
            f"write={self._t_write_total:.2f}s  "
            f"close={t_close:.2f}s  "
            f"total={total_elapsed:.2f}s  "
            f"→ {self._output_path}",
            flush=True,
        )
        logger.info(
            "Animation finalized: %d frames, render=%.2fs, write=%.2fs, "
            "close=%.2fs, output=%s",
            self._total_frames, self._t_render_total, self._t_write_total,
            t_close, self._output_path,
        )
