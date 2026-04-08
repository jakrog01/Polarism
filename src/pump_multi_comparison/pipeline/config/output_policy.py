from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class OutputPolicy:
    field_record_stride: int = 100
    scalar_record_stride: int = 10
    render_snapshots: bool = True
    render_animation: bool = True
    archive_raw_hdf5: bool = False


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
    return OutputPolicy(
        field_record_stride=field_stride,
        scalar_record_stride=scalar_stride,
        render_snapshots=bool(out.get("render_snapshots", True)),
        render_animation=bool(out.get("render_animation", True)),
        archive_raw_hdf5=bool(out.get("archive_raw_hdf5", False)),
    )
