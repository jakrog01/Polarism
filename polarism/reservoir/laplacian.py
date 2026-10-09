"""Reservoir-field Laplacian helpers."""
from __future__ import annotations

from typing import Any


def real_laplacian(field: Any, xp: Any, dx: float, dy: float, grid_type: str) -> Any:
    """Return the finite-difference Laplacian of a real reservoir field.

    Periodic grids wrap at the boundary.  Closed-interval grids use the same
    mirrored Neumann boundary convention as the finite-difference psi solvers.
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    if grid_type == "periodic":
        return (
            xp.roll(field, -1, axis=-2) - 2 * field + xp.roll(field, 1, axis=-2)
        ) * inv_dy2 + (
            xp.roll(field, -1, axis=-1) - 2 * field + xp.roll(field, 1, axis=-1)
        ) * inv_dx2

    if grid_type != "closed-interval":
        raise ValueError(f"Unknown grid_type for reservoir diffusion: {grid_type}")

    ny, nx = field.shape[-2:]
    out = xp.empty_like(field)
    out.fill(0)

    if ny > 2 and nx > 2:
        out[..., 1:-1, 1:-1] = (
            field[..., 2:, 1:-1] - 2 * field[..., 1:-1, 1:-1] + field[..., :-2, 1:-1]
        ) * inv_dy2 + (
            field[..., 1:-1, 2:] - 2 * field[..., 1:-1, 1:-1] + field[..., 1:-1, :-2]
        ) * inv_dx2

    if nx > 2:
        out[..., 0, 1:-1] = (
            2 * (field[..., 1, 1:-1] - field[..., 0, 1:-1]) * inv_dy2
            + (field[..., 0, 2:] - 2 * field[..., 0, 1:-1] + field[..., 0, :-2]) * inv_dx2
        )
        out[..., -1, 1:-1] = (
            2 * (field[..., -2, 1:-1] - field[..., -1, 1:-1]) * inv_dy2
            + (field[..., -1, 2:] - 2 * field[..., -1, 1:-1] + field[..., -1, :-2]) * inv_dx2
        )

    if ny > 2:
        out[..., 1:-1, 0] = (
            field[..., 2:, 0] - 2 * field[..., 1:-1, 0] + field[..., :-2, 0]
        ) * inv_dy2 + 2 * (field[..., 1:-1, 1] - field[..., 1:-1, 0]) * inv_dx2
        out[..., 1:-1, -1] = (
            field[..., 2:, -1] - 2 * field[..., 1:-1, -1] + field[..., :-2, -1]
        ) * inv_dy2 + 2 * (field[..., 1:-1, -2] - field[..., 1:-1, -1]) * inv_dx2

    out[..., 0, 0] = (
        2 * (field[..., 1, 0] - field[..., 0, 0]) * inv_dy2
        + 2 * (field[..., 0, 1] - field[..., 0, 0]) * inv_dx2
    )
    out[..., 0, -1] = (
        2 * (field[..., 1, -1] - field[..., 0, -1]) * inv_dy2
        + 2 * (field[..., 0, -2] - field[..., 0, -1]) * inv_dx2
    )
    out[..., -1, 0] = (
        2 * (field[..., -2, 0] - field[..., -1, 0]) * inv_dy2
        + 2 * (field[..., -1, 1] - field[..., -1, 0]) * inv_dx2
    )
    out[..., -1, -1] = (
        2 * (field[..., -2, -1] - field[..., -1, -1]) * inv_dy2
        + 2 * (field[..., -1, -2] - field[..., -1, -1]) * inv_dx2
    )
    return out
