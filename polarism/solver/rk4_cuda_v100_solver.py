"""V100-tuned RK4 CUDA solver."""
from __future__ import annotations

import numpy as _np

from polarism.solver.rk4_cuda_solver import (
    RK4CudaSolver,
    build_kernel_source_1d,
)
from polarism.solver.solver_registry import register_solver

_PREAMBLE_2D = """
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;
    const int tid = i * nx + j;
"""

_COMBINE_PREAMBLE_2D = """
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= ny || j >= nx) return;
    const int tid = i * nx + j;
"""

_LAUNCH_BOUNDS_V100 = "__launch_bounds__(256, 4)"
_BLOCK_2D = (32, 8)


def _build_kernel_source_v100(
    bc_type: str, real_type: str, reservoir_type: str, laplacian_type: str = "five-point"
) -> tuple[str, str, str]:
    """Build the V100 CUDA kernel source."""
    return build_kernel_source_1d(
        bc_type,
        real_type,
        reservoir_type,
        laplacian_type=laplacian_type,
        preamble=_PREAMBLE_2D,
        combine_preamble=_COMBINE_PREAMBLE_2D,
        launch_bounds=_LAUNCH_BOUNDS_V100,
    )

@register_solver("rk4-cuda-v100")
class RK4CudaV100Solver(RK4CudaSolver):
    """Select via `cfg.solver.method = "rk4-cuda-v100"`."""

    def _build_kernel_source(
        self, bc_type: str, real_type: str, reservoir_type: str
    ) -> tuple[str, str, str]:
        """Build the CUDA kernel source."""
        return _build_kernel_source_v100(bc_type, real_type, reservoir_type, self._laplacian_type)

    def _select_block_size(self) -> tuple[int, int]:
        """Pick the CUDA block size."""
        return _BLOCK_2D

    def _make_launch_dims(self, block_size: tuple[int, int]) -> None:
        """Set the CUDA launch dimensions."""
        bx, by = block_size
        gx = (self.nx + bx - 1) // bx
        gy = (self.ny + by - 1) // by
        self._k_grid = (gx, gy)
        self._k_block = (bx, by)
