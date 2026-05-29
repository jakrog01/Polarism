from __future__ import annotations

import dataclasses
import warnings


@dataclasses.dataclass
class OutputPolicy:
    field_record_stride: int = 100
    scalar_record_stride: int = 10
    render_snapshots: bool = True
    render_animation: bool = True
    archive_raw_hdf5: bool = False
    render_downscale_factor: int = 1
    animation_clim: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
    fringe_analysis_enabled: bool = False
    fringe_analysis_window_radius: float = 16.0


def output_policy_from_config(cfg: dict) -> OutputPolicy:
    """Build an OutputPolicy from the ``output`` section of a pipeline config."""
    out = cfg.get("output", {})
    field_stride = int(out.get("field_record_stride", 100))
    scalar_stride = int(out.get("scalar_record_stride", 10))
    if field_stride < 1:
        raise ValueError(
            f"output.field_record_stride must be >= 1, got {field_stride}"
        )
    if scalar_stride < 1:
        raise ValueError(
            f"output.scalar_record_stride must be >= 1, got {scalar_stride}"
        )

    raw_clim = out.get("animation_clim", {})
    animation_clim: dict[str, tuple[float, float]] = {}
    for field_key, bounds in raw_clim.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            animation_clim[str(field_key)] = (float(bounds[0]), float(bounds[1]))
        else:
            warnings.warn(
                f"output.animation_clim[{field_key!r}] must be a 2-element list [vmin, vmax]; "
                f"got {bounds!r} — entry ignored.",
                UserWarning,
                stacklevel=2,
            )

    downscale = int(out.get("render_downscale_factor", 1))
    if downscale < 1:
        raise ValueError(
            f"output.render_downscale_factor must be >= 1, got {downscale}"
        )

    fringe_cfg = out.get("fringe_analysis", {})
    fringe_enabled = bool(fringe_cfg.get("enabled", False))
    fringe_radius = float(fringe_cfg.get("fringe_window_radius", 16.0))

    return OutputPolicy(
        field_record_stride=field_stride,
        scalar_record_stride=scalar_stride,
        render_snapshots=bool(out.get("render_snapshots", True)),
        render_animation=bool(out.get("render_animation", True)),
        archive_raw_hdf5=bool(out.get("archive_raw_hdf5", False)),
        render_downscale_factor=downscale,
        animation_clim=animation_clim,
        fringe_analysis_enabled=fringe_enabled,
        fringe_analysis_window_radius=fringe_radius,
    )
