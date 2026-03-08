from __future__ import annotations

from typing import TYPE_CHECKING, Union

from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.grid.simulation_grid_2d import SimulationGrid2D
    from polarism.reservoir.abstract_reservoir import AbstractReservoir


# ---------------------------------------------------------------------------
# CUDA kernels: fused GP equation RHS + reservoir derivatives
#
# Three kernel types per reservoir variant:
#   1. gpe_rhs       — standalone RHS (used for k1)
#   2. fused_stage_rhs — compute intermediate state + RHS in one pass (k2-k4)
#   3. rk4_combine   — final weighted combination of k1-k4
#
# This gives 5 kernel launches per time step (vs 8 before fusion).
# ---------------------------------------------------------------------------

# ---- Neighbor code snippets ----

_NEUMANN_NEIGHBORS = """
    // Neumann mirror BC: ghost point reflects across boundary
    const int up = (i > 0)    ? tid - nx : tid + nx;
    const int dn = (i < ny-1) ? tid + nx : tid - nx;
    const int lt = (j > 0)    ? tid - 1  : tid + 1;
    const int rt = (j < nx-1) ? tid + 1  : tid - 1;
"""

_PERIODIC_NEIGHBORS = """
    // Periodic BC: wrap around
    const int up = ((i > 0)    ? i - 1 : ny - 1) * nx + j;
    const int dn = ((i < ny-1) ? i + 1 : 0)      * nx + j;
    const int lt = i * nx + ((j > 0)    ? j - 1 : nx - 1);
    const int rt = i * nx + ((j < nx-1) ? j + 1 : 0);
"""

# ---------------------------------------------------------------------------
# Single reservoir kernels
# ---------------------------------------------------------------------------

_KERNEL_RHS_TEMPLATE = r"""
extern "C" __global__ void gpe_rhs(
    const {REAL}* __restrict__ psi,
    const {REAL}* __restrict__ nR,
    const {REAL}* __restrict__ V,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    const int nx, const int ny,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C, const {REAL} gamma_R
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;

    const int i = tid / nx;
    const int j = tid % nx;

    const {REAL} pre = psi[2*tid];
    const {REAL} pim = psi[2*tid + 1];

    {NEIGHBOR_CODE}

    const {REAL} lap_re = (psi[2*dn]   - 2*pre + psi[2*up])   * inv_dy2
                        + (psi[2*rt]   - 2*pre + psi[2*lt])   * inv_dx2;
    const {REAL} lap_im = (psi[2*dn+1] - 2*pim + psi[2*up+1]) * inv_dy2
                        + (psi[2*rt+1] - 2*pim + psi[2*lt+1]) * inv_dx2;

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} nR_val = nR[tid];
    const {REAL} eff = V[tid] + g_C * rho + g_R * nR_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim;

    dnR_dt[tid] = pump[tid] - (gamma_R + R_cond * rho) * nR_val;
}}
"""

_KERNEL_FUSED_STAGE_RHS_TEMPLATE = r"""
extern "C" __global__ void fused_stage_rhs(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k_nR,
    const {REAL}* __restrict__ V,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    const int nx, const int ny,
    const {REAL} c_dt,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C, const {REAL} gamma_R
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;

    // Stage: compute intermediate state in registers (no global write)
    const {REAL} pre = psi0[2*tid]   + c_dt * k_psi[2*tid];
    const {REAL} pim = psi0[2*tid+1] + c_dt * k_psi[2*tid+1];
    const {REAL} nR_val = nR0[tid] + c_dt * k_nR[tid];

    const int i = tid / nx;
    const int j = tid % nx;

    {NEIGHBOR_CODE}

    // Must read neighbors from the staged values too.
    // We cannot keep all neighbors in registers, so we recompute them.
    const {REAL} psi_up_re = psi0[2*up]   + c_dt * k_psi[2*up];
    const {REAL} psi_up_im = psi0[2*up+1] + c_dt * k_psi[2*up+1];
    const {REAL} psi_dn_re = psi0[2*dn]   + c_dt * k_psi[2*dn];
    const {REAL} psi_dn_im = psi0[2*dn+1] + c_dt * k_psi[2*dn+1];
    const {REAL} psi_lt_re = psi0[2*lt]   + c_dt * k_psi[2*lt];
    const {REAL} psi_lt_im = psi0[2*lt+1] + c_dt * k_psi[2*lt+1];
    const {REAL} psi_rt_re = psi0[2*rt]   + c_dt * k_psi[2*rt];
    const {REAL} psi_rt_im = psi0[2*rt+1] + c_dt * k_psi[2*rt+1];

    const {REAL} lap_re = (psi_dn_re - 2*pre + psi_up_re) * inv_dy2
                        + (psi_rt_re - 2*pre + psi_lt_re) * inv_dx2;
    const {REAL} lap_im = (psi_dn_im - 2*pim + psi_up_im) * inv_dy2
                        + (psi_rt_im - 2*pim + psi_lt_im) * inv_dx2;

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} eff = V[tid] + g_C * rho + g_R * nR_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim;

    dnR_dt[tid] = pump[tid] - (gamma_R + R_cond * rho) * nR_val;
}}
"""

_KERNEL_COMBINE = r"""
extern "C" __global__ void rk4_combine(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k1p, const {REAL}* __restrict__ k2p,
    const {REAL}* __restrict__ k3p, const {REAL}* __restrict__ k4p,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k1n, const {REAL}* __restrict__ k2n,
    const {REAL}* __restrict__ k3n, const {REAL}* __restrict__ k4n,
    {REAL}* __restrict__ psi_out,
    {REAL}* __restrict__ nR_out,
    const {REAL} dt6,
    const int N
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k3p[2*tid]   + k4p[2*tid]);
    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k3p[2*tid+1] + k4p[2*tid+1]);
    {REAL} nR_new = nR0[tid] + dt6 * (k1n[tid] + 2*k2n[tid] + 2*k3n[tid] + k4n[tid]);
    nR_out[tid] = (nR_new > 0) ? nR_new : 0;
}}
"""

# ---------------------------------------------------------------------------
# Double reservoir kernels
# ---------------------------------------------------------------------------

_KERNEL_RHS_DOUBLE_TEMPLATE = r"""
extern "C" __global__ void gpe_rhs_double(
    const {REAL}* __restrict__ psi,
    const {REAL}* __restrict__ nA,
    const {REAL}* __restrict__ nI,
    const {REAL}* __restrict__ V,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnA_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_I, const {REAL} gamma_A,
    const {REAL} R_IA, const {REAL} R_AI
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;

    const int i = tid / nx;
    const int j = tid % nx;

    const {REAL} pre = psi[2*tid];
    const {REAL} pim = psi[2*tid + 1];

    {NEIGHBOR_CODE}

    const {REAL} lap_re = (psi[2*dn]   - 2*pre + psi[2*up])   * inv_dy2
                        + (psi[2*rt]   - 2*pre + psi[2*lt])   * inv_dx2;
    const {REAL} lap_im = (psi[2*dn+1] - 2*pim + psi[2*up+1]) * inv_dy2
                        + (psi[2*rt+1] - 2*pim + psi[2*lt+1]) * inv_dx2;

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} nA_val = nA[tid];
    const {REAL} nI_val = nI[tid];

    const {REAL} eff = V[tid] + g_C * rho + g_R * nA_val;
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim;

    dnI_dt[tid] = pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val;
    dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val;
}}
"""

_KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE = r"""
extern "C" __global__ void fused_stage_rhs_double(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nA0,
    const {REAL}* __restrict__ k_nA,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k_nI,
    const {REAL}* __restrict__ V,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnA_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} c_dt,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_I, const {REAL} gamma_A,
    const {REAL} R_IA, const {REAL} R_AI
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;

    // Stage: intermediate state in registers
    const {REAL} pre = psi0[2*tid]   + c_dt * k_psi[2*tid];
    const {REAL} pim = psi0[2*tid+1] + c_dt * k_psi[2*tid+1];
    const {REAL} nA_val = nA0[tid] + c_dt * k_nA[tid];
    const {REAL} nI_val = nI0[tid] + c_dt * k_nI[tid];

    const int i = tid / nx;
    const int j = tid % nx;

    {NEIGHBOR_CODE}

    const {REAL} psi_up_re = psi0[2*up]   + c_dt * k_psi[2*up];
    const {REAL} psi_up_im = psi0[2*up+1] + c_dt * k_psi[2*up+1];
    const {REAL} psi_dn_re = psi0[2*dn]   + c_dt * k_psi[2*dn];
    const {REAL} psi_dn_im = psi0[2*dn+1] + c_dt * k_psi[2*dn+1];
    const {REAL} psi_lt_re = psi0[2*lt]   + c_dt * k_psi[2*lt];
    const {REAL} psi_lt_im = psi0[2*lt+1] + c_dt * k_psi[2*lt+1];
    const {REAL} psi_rt_re = psi0[2*rt]   + c_dt * k_psi[2*rt];
    const {REAL} psi_rt_im = psi0[2*rt+1] + c_dt * k_psi[2*rt+1];

    const {REAL} lap_re = (psi_dn_re - 2*pre + psi_up_re) * inv_dy2
                        + (psi_rt_re - 2*pre + psi_lt_re) * inv_dx2;
    const {REAL} lap_im = (psi_dn_im - 2*pim + psi_up_im) * inv_dy2
                        + (psi_rt_im - 2*pim + psi_lt_im) * inv_dx2;

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} eff = V[tid] + g_C * rho + g_R * nA_val;
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim;

    dnI_dt[tid] = pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val;
    dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val;
}}
"""

_KERNEL_COMBINE_DOUBLE = r"""
extern "C" __global__ void rk4_combine_double(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k1p, const {REAL}* __restrict__ k2p,
    const {REAL}* __restrict__ k3p, const {REAL}* __restrict__ k4p,
    const {REAL}* __restrict__ nA0,
    const {REAL}* __restrict__ k1a, const {REAL}* __restrict__ k2a,
    const {REAL}* __restrict__ k3a, const {REAL}* __restrict__ k4a,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k1i, const {REAL}* __restrict__ k2i,
    const {REAL}* __restrict__ k3i, const {REAL}* __restrict__ k4i,
    {REAL}* __restrict__ psi_out,
    {REAL}* __restrict__ nA_out,
    {REAL}* __restrict__ nI_out,
    const {REAL} dt6,
    const int N
) {{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k3p[2*tid]   + k4p[2*tid]);
    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k3p[2*tid+1] + k4p[2*tid+1]);
    {REAL} nA_new = nA0[tid] + dt6 * (k1a[tid] + 2*k2a[tid] + 2*k3a[tid] + k4a[tid]);
    nA_out[tid] = (nA_new > 0) ? nA_new : 0;
    {REAL} nI_new = nI0[tid] + dt6 * (k1i[tid] + 2*k2i[tid] + 2*k3i[tid] + k4i[tid]);
    nI_out[tid] = (nI_new > 0) ? nI_new : 0;
}}
"""


def _build_kernel_code(
    bc_type: str, real_type: str = "double", reservoir_type: str = "single"
) -> tuple[str, str, str]:
    if bc_type == "closed-interval":
        neighbor_code = _NEUMANN_NEIGHBORS
        bc_comment = "Neumann mirror"
    else:
        neighbor_code = _PERIODIC_NEIGHBORS
        bc_comment = "Periodic wrap"

    if reservoir_type == "double":
        rhs_tmpl = _KERNEL_RHS_DOUBLE_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE
        combine_tmpl = _KERNEL_COMBINE_DOUBLE
    else:
        rhs_tmpl = _KERNEL_RHS_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_TEMPLATE
        combine_tmpl = _KERNEL_COMBINE

    fmt = dict(REAL=real_type, NEIGHBOR_CODE=neighbor_code, BC_COMMENT=bc_comment)

    rhs_code = rhs_tmpl.format(**fmt)
    fused_code = fused_tmpl.format(**fmt)
    combine_code = combine_tmpl.format(**fmt)
    return rhs_code, fused_code, combine_code


def _auto_tune_block_size(kern_rhs, _N: int) -> int:
    """Pick optimal block size from candidates by querying occupancy."""
    candidates = [128, 256, 512]
    best_block = 256
    best_occupancy = 0
    for bs in candidates:
        try:
            max_active = kern_rhs.max_active_blocks_per_multiprocessor(bs)
            occupancy = max_active * bs
            if occupancy > best_occupancy:
                best_occupancy = occupancy
                best_block = bs
        except Exception:
            continue
    return best_block


@register_solver("rk4-cuda")
class RK4CudaSolver(AbstractSolver):
    def __init__(self, config: Config, grid: SimulationGrid2D):
        super().__init__(config)
        self.nx = grid.nx
        self.ny = grid.ny
        self.N = grid.nx * grid.ny
        self.grid_type = getattr(config.grid, "grid_type", "periodic")
        _res_cfg = getattr(config, "reservoir", None)
        self._reservoir_type = getattr(_res_cfg, "reservoir_type", "single") if _res_cfg else "single"

        # Precision config
        import numpy as _np

        self._precision = getattr(config.solver, "precision", "double")
        if self._precision == "single":
            self._real_type = "float"
            self._np_real = _np.float32
            self._np_complex = _np.complex64
        else:
            self._real_type = "double"
            self._np_real = _np.float64
            self._np_complex = _np.complex128

        self._hbar = config.physics.hbar
        self._m_eff = config.physics.m_eff
        self._g_C = config.physics.g_C
        self._g_R = config.physics.g_R
        self._R = config.physics.R
        self._gamma_C = config.physics.gamma_C
        self._gamma_R = config.physics.gamma_R
        self._dt = config.solver.dt

        if self._reservoir_type == "double":
            self._gamma_I = config.physics.gamma_I
            self._gamma_A = config.physics.gamma_A
            self._R_IA = config.physics.R_IA
            self._R_AI = config.physics.R_AI

        self._inv_dx2 = 1.0 / (grid.dx**2)
        self._inv_dy2 = 1.0 / (grid.dy**2)
        self._neg_hbar2_over_2m = -(self._hbar**2) / (2.0 * self._m_eff)
        self._inv_hbar = 1.0 / self._hbar

        self._use_gpu = hasattr(self.xp, "cuda")
        self._buf_initialized = False

        if self._use_gpu:
            _cast = self._np_real

            self._k_nx = _np.int32(self.nx)
            self._k_ny = _np.int32(self.ny)
            self._k_N = _np.int32(self.N)
            self._k_inv_dx2 = _cast(self._inv_dx2)
            self._k_inv_dy2 = _cast(self._inv_dy2)
            self._k_neg_hbar2_over_2m = _cast(self._neg_hbar2_over_2m)
            self._k_inv_hbar = _cast(self._inv_hbar)
            self._k_g_C = _cast(self._g_C)
            self._k_g_R = _cast(self._g_R)
            self._k_R = _cast(self._R)
            self._k_gamma_C = _cast(self._gamma_C)
            self._k_gamma_R = _cast(self._gamma_R)

            if self._reservoir_type == "double":
                self._k_gamma_I = _cast(self._gamma_I)
                self._k_gamma_A = _cast(self._gamma_A)
                self._k_R_IA = _cast(self._R_IA)
                self._k_R_AI = _cast(self._R_AI)

            self._compile_kernels()

            # CUDA streams for async save overlap
            import cupy as _cp
            self._stream_compute = _cp.cuda.Stream(non_blocking=True)
            self._stream_transfer = _cp.cuda.Stream(non_blocking=True)
            self._transfer_event = _cp.cuda.Event()

    def _compile_kernels(self):
        import cupy as cp

        rhs_src, fused_src, combine_src = _build_kernel_code(
            self.grid_type,
            real_type=self._real_type,
            reservoir_type=self._reservoir_type,
        )

        if self._reservoir_type == "double":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_double")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_double")
            self._kern_combine = cp.RawKernel(combine_src, "rk4_combine_double")
        else:
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs")
            self._kern_combine = cp.RawKernel(combine_src, "rk4_combine")

        # Auto-tune block size
        self._block = _auto_tune_block_size(self._kern_rhs, self.N)
        self._grid_rhs = (self.N + self._block - 1) // self._block

    def _ensure_buffers(self, psi=None):
        if self._buf_initialized:
            return
        xp = self.xp
        N = self.N

        if self._precision == "single":
            cdtype = xp.complex64
            rdtype = xp.float32
        else:
            cdtype = xp.complex128
            rdtype = xp.float64

        self._k1_psi = xp.empty(N, dtype=cdtype)
        self._k2_psi = xp.empty(N, dtype=cdtype)
        self._k3_psi = xp.empty(N, dtype=cdtype)
        self._k4_psi = xp.empty(N, dtype=cdtype)

        self._k1_nA = xp.empty(N, dtype=rdtype)
        self._k2_nA = xp.empty(N, dtype=rdtype)
        self._k3_nA = xp.empty(N, dtype=rdtype)
        self._k4_nA = xp.empty(N, dtype=rdtype)

        self._psi_tmp = xp.empty(N, dtype=cdtype)
        self._nA_tmp = xp.empty(N, dtype=rdtype)

        self._psi0 = xp.empty(N, dtype=cdtype)
        self._nA0 = xp.empty(N, dtype=rdtype)

        if self._reservoir_type == "double":
            self._k1_nI = xp.empty(N, dtype=rdtype)
            self._k2_nI = xp.empty(N, dtype=rdtype)
            self._k3_nI = xp.empty(N, dtype=rdtype)
            self._k4_nI = xp.empty(N, dtype=rdtype)
            self._nI_tmp = xp.empty(N, dtype=rdtype)
            self._nI0 = xp.empty(N, dtype=rdtype)

        self._buf_initialized = True

    def _cast_to_precision(self, arr):
        """Cast array to solver precision if needed."""
        xp = self.xp
        if self._precision == "single":
            if xp.iscomplexobj(arr):
                if arr.dtype != xp.complex64:
                    return arr.astype(xp.complex64)
            else:
                if arr.dtype != xp.float32:
                    return arr.astype(xp.float32)
        return arr

    # ------------------------------------------------------------------
    # Single reservoir GPU path (fused: 5 launches per step)
    # ------------------------------------------------------------------

    def _gpu_rhs(self, psi, nR, V, pump, out_dpsi, out_dnR):
        self._kern_rhs(
            (self._grid_rhs,),
            (self._block,),
            (
                psi, nR, V, pump, out_dpsi, out_dnR,
                self._k_nx, self._k_ny,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C, self._k_gamma_R,
            ),
        )

    def _gpu_fused_stage_rhs(self, psi0, k_psi, nR0, k_nR, c_dt, V, pump, out_dpsi, out_dnR):
        self._kern_fused(
            (self._grid_rhs,),
            (self._block,),
            (
                psi0, k_psi, nR0, k_nR, V, pump, out_dpsi, out_dnR,
                self._k_nx, self._k_ny,
                self._np_real(c_dt),
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C, self._k_gamma_R,
            ),
        )

    def _gpu_combine(
        self, psi0, k1p, k2p, k3p, k4p, nR0, k1n, k2n, k3n, k4n, psi_out, nR_out
    ):
        self._kern_combine(
            (self._grid_rhs,),
            (self._block,),
            (
                psi0, k1p, k2p, k3p, k4p,
                nR0, k1n, k2n, k3n, k4n,
                psi_out, nR_out,
                self._np_real(self._dt / 6.0),
                self._k_N,
            ),
        )

    def _step_gpu(self, V, pump, nR_buf, psi_buf):
        dt = self._dt
        psi0 = self._psi0
        nR0 = self._nA0

        psi0[:] = psi_buf
        nR0[:] = nR_buf

        # k1: standalone RHS on initial state
        self._gpu_rhs(psi0, nR0, V, pump, self._k1_psi, self._k1_nA)

        # k2: fused stage(psi0 + 0.5*dt*k1) + RHS
        self._gpu_fused_stage_rhs(
            psi0, self._k1_psi, nR0, self._k1_nA,
            0.5 * dt, V, pump, self._k2_psi, self._k2_nA,
        )

        # k3: fused stage(psi0 + 0.5*dt*k2) + RHS
        self._gpu_fused_stage_rhs(
            psi0, self._k2_psi, nR0, self._k2_nA,
            0.5 * dt, V, pump, self._k3_psi, self._k3_nA,
        )

        # k4: fused stage(psi0 + dt*k3) + RHS
        self._gpu_fused_stage_rhs(
            psi0, self._k3_psi, nR0, self._k3_nA,
            dt, V, pump, self._k4_psi, self._k4_nA,
        )

        # Combine: 5th launch
        self._gpu_combine(
            psi0, self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi,
            nR0, self._k1_nA, self._k2_nA, self._k3_nA, self._k4_nA,
            psi_buf, nR_buf,
        )

    # ------------------------------------------------------------------
    # Double reservoir GPU path (fused: 5 launches per step)
    # ------------------------------------------------------------------

    def _gpu_rhs_double(self, psi, nA, nI, V, pump, out_dpsi, out_dnA, out_dnI):
        self._kern_rhs(
            (self._grid_rhs,),
            (self._block,),
            (
                psi, nA, nI, V, pump, out_dpsi, out_dnA, out_dnI,
                self._k_nx, self._k_ny,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C,
                self._k_gamma_I, self._k_gamma_A,
                self._k_R_IA, self._k_R_AI,
            ),
        )

    def _gpu_fused_stage_rhs_double(
        self, psi0, k_psi, nA0, k_nA, nI0, k_nI, c_dt, V, pump,
        out_dpsi, out_dnA, out_dnI,
    ):
        self._kern_fused(
            (self._grid_rhs,),
            (self._block,),
            (
                psi0, k_psi, nA0, k_nA, nI0, k_nI, V, pump,
                out_dpsi, out_dnA, out_dnI,
                self._k_nx, self._k_ny,
                self._np_real(c_dt),
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C,
                self._k_gamma_I, self._k_gamma_A,
                self._k_R_IA, self._k_R_AI,
            ),
        )

    def _gpu_combine_double(
        self, psi0, k1p, k2p, k3p, k4p,
        nA0, k1a, k2a, k3a, k4a,
        nI0, k1i, k2i, k3i, k4i,
        psi_out, nA_out, nI_out,
    ):
        self._kern_combine(
            (self._grid_rhs,),
            (self._block,),
            (
                psi0, k1p, k2p, k3p, k4p,
                nA0, k1a, k2a, k3a, k4a,
                nI0, k1i, k2i, k3i, k4i,
                psi_out, nA_out, nI_out,
                self._np_real(self._dt / 6.0),
                self._k_N,
            ),
        )

    def _step_gpu_double(self, V, pump, nA_buf, nI_buf, psi_buf):
        dt = self._dt
        psi0 = self._psi0
        nA0 = self._nA0
        nI0 = self._nI0

        psi0[:] = psi_buf
        nA0[:] = nA_buf
        nI0[:] = nI_buf

        # k1
        self._gpu_rhs_double(psi0, nA0, nI0, V, pump,
                             self._k1_psi, self._k1_nA, self._k1_nI)

        # k2: fused
        self._gpu_fused_stage_rhs_double(
            psi0, self._k1_psi, nA0, self._k1_nA, nI0, self._k1_nI,
            0.5 * dt, V, pump,
            self._k2_psi, self._k2_nA, self._k2_nI,
        )

        # k3: fused
        self._gpu_fused_stage_rhs_double(
            psi0, self._k2_psi, nA0, self._k2_nA, nI0, self._k2_nI,
            0.5 * dt, V, pump,
            self._k3_psi, self._k3_nA, self._k3_nI,
        )

        # k4: fused
        self._gpu_fused_stage_rhs_double(
            psi0, self._k3_psi, nA0, self._k3_nA, nI0, self._k3_nI,
            dt, V, pump,
            self._k4_psi, self._k4_nA, self._k4_nI,
        )

        # Combine
        self._gpu_combine_double(
            psi0,
            self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi,
            nA0,
            self._k1_nA, self._k2_nA, self._k3_nA, self._k4_nA,
            nI0,
            self._k1_nI, self._k2_nI, self._k3_nI, self._k4_nI,
            psi_buf, nA_buf, nI_buf,
        )

    # ------------------------------------------------------------------
    # CPU fallback paths (unchanged)
    # ------------------------------------------------------------------

    def _cpu_laplacian(self, psi, out):
        xp = self.xp
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        if self.grid_type == "periodic":
            out[:] = (
                xp.roll(psi, -1, axis=0) - 2 * psi + xp.roll(psi, 1, axis=0)
            ) * inv_dy2 + (
                xp.roll(psi, -1, axis=1) - 2 * psi + xp.roll(psi, 1, axis=1)
            ) * inv_dx2
            return

        ny, nx = self.ny, self.nx
        out.fill(0)
        if ny > 2 and nx > 2:
            out[1:-1, 1:-1] = (
                psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]
            ) * inv_dy2 + (
                psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]
            ) * inv_dx2
        if nx > 2:
            out[0, 1:-1] = (
                2 * (psi[1, 1:-1] - psi[0, 1:-1]) * inv_dy2
                + (psi[0, 2:] - 2 * psi[0, 1:-1] + psi[0, :-2]) * inv_dx2
            )
            out[-1, 1:-1] = (
                2 * (psi[-2, 1:-1] - psi[-1, 1:-1]) * inv_dy2
                + (psi[-1, 2:] - 2 * psi[-1, 1:-1] + psi[-1, :-2]) * inv_dx2
            )
        if ny > 2:
            out[1:-1, 0] = (
                psi[2:, 0] - 2 * psi[1:-1, 0] + psi[:-2, 0]
            ) * inv_dy2 + 2 * (psi[1:-1, 1] - psi[1:-1, 0]) * inv_dx2
            out[1:-1, -1] = (
                psi[2:, -1] - 2 * psi[1:-1, -1] + psi[:-2, -1]
            ) * inv_dy2 + 2 * (psi[1:-1, -2] - psi[1:-1, -1]) * inv_dx2
        out[0, 0] = (
            2 * (psi[1, 0] - psi[0, 0]) * inv_dy2
            + 2 * (psi[0, 1] - psi[0, 0]) * inv_dx2
        )
        out[0, -1] = (
            2 * (psi[1, -1] - psi[0, -1]) * inv_dy2
            + 2 * (psi[0, -2] - psi[0, -1]) * inv_dx2
        )
        out[-1, 0] = (
            2 * (psi[-2, 0] - psi[-1, 0]) * inv_dy2
            + 2 * (psi[-1, 1] - psi[-1, 0]) * inv_dx2
        )
        out[-1, -1] = (
            2 * (psi[-2, -1] - psi[-1, -1]) * inv_dy2
            + 2 * (psi[-1, -2] - psi[-1, -1]) * inv_dx2
        )

    def _cpu_psi_rhs(self, psi, n_active, V):
        xp = self.xp
        lap = xp.empty_like(psi)
        self._cpu_laplacian(psi, lap)
        rho = xp.abs(psi) ** 2
        eff = V + self._g_C * rho + self._g_R * n_active
        gl = (self._R * n_active - self._gamma_C) * 0.5
        kinetic = self._neg_hbar2_over_2m * lap
        return (-1j * self._inv_hbar) * (kinetic + eff * psi) + gl * psi

    def _cpu_rhs(self, psi, nR, V, pump):
        dpsi = self._cpu_psi_rhs(psi, nR, V)
        rho = self.xp.abs(psi) ** 2
        dnR = pump - (self._gamma_R + self._R * rho) * nR
        return dpsi, dnR

    def _cpu_rhs_double(self, psi, nA, nI, V, pump):
        dpsi = self._cpu_psi_rhs(psi, nA, V)
        rho = self.xp.abs(psi) ** 2
        dnI = pump - (self._gamma_I + self._R_IA) * nI + self._R_AI * nA
        dnA = self._R_IA * nI - (self._gamma_A + self._R_AI + self._R * rho) * nA
        return dpsi, dnA, dnI

    def _step_cpu(self, V, pump, reservoir, boundary_condition, state):
        dt = self._dt
        psi0 = state.psi.copy()
        nR0 = reservoir.get_reservoir_density().copy()

        k1p, k1n = self._cpu_rhs(psi0, nR0, V, pump)

        psi2 = psi0 + 0.5 * dt * k1p
        nR2 = nR0 + 0.5 * dt * k1n
        k2p, k2n = self._cpu_rhs(psi2, nR2, V, pump)

        psi3 = psi0 + 0.5 * dt * k2p
        nR3 = nR0 + 0.5 * dt * k2n
        k3p, k3n = self._cpu_rhs(psi3, nR3, V, pump)

        psi4 = psi0 + dt * k3p
        nR4 = nR0 + dt * k3n
        k4p, k4n = self._cpu_rhs(psi4, nR4, V, pump)

        state.psi = psi0 + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        state.psi = boundary_condition.after_step_action(state.psi)
        nR_new = nR0 + (dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n)
        reservoir.set_state((nR_new,))

    def _step_cpu_double(self, V, pump, reservoir, boundary_condition, state):
        dt = self._dt
        xp = self.xp
        psi0 = state.psi.copy()
        nA0, nI0 = reservoir.get_state()
        nA0 = nA0.copy()
        nI0 = nI0.copy()

        k1p, k1a, k1i = self._cpu_rhs_double(psi0, nA0, nI0, V, pump)

        k2p, k2a, k2i = self._cpu_rhs_double(
            psi0 + 0.5 * dt * k1p, nA0 + 0.5 * dt * k1a, nI0 + 0.5 * dt * k1i, V, pump
        )
        k3p, k3a, k3i = self._cpu_rhs_double(
            psi0 + 0.5 * dt * k2p, nA0 + 0.5 * dt * k2a, nI0 + 0.5 * dt * k2i, V, pump
        )
        k4p, k4a, k4i = self._cpu_rhs_double(
            psi0 + dt * k3p, nA0 + dt * k3a, nI0 + dt * k3i, V, pump
        )

        dt6 = dt / 6.0
        state.psi = psi0 + dt6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        state.psi = boundary_condition.after_step_action(state.psi)
        nA_new = nA0 + dt6 * (k1a + 2 * k2a + 2 * k3a + k4a)
        nI_new = nI0 + dt6 * (k1i + 2 * k2i + 2 * k3i + k4i)
        reservoir.set_state((xp.maximum(nA_new, 0), xp.maximum(nI_new, 0)))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        if not self._use_gpu:
            if self._reservoir_type == "double":
                self._step_cpu_double(potential, pump, reservoir, boundary_condition, state)
            else:
                self._step_cpu(potential, pump, reservoir, boundary_condition, state)
            return

        self._ensure_buffers(state.psi)
        shape = state.psi.shape

        # Cast inputs to solver precision
        V_flat = self._cast_to_precision(potential).ravel()
        pump_flat = self._cast_to_precision(pump).ravel()

        with self._stream_compute:
            if self._reservoir_type == "double":
                nA, nI = reservoir.get_state()
                nA_flat = self._cast_to_precision(nA).ravel()
                nI_flat = self._cast_to_precision(nI).ravel()
                psi_buf = self._psi_tmp
                psi_buf[:] = self._cast_to_precision(state.psi).ravel()
                nA_buf = self._nA_tmp
                nA_buf[:] = nA_flat
                nI_buf = self._nI_tmp
                nI_buf[:] = nI_flat

                self._step_gpu_double(V_flat, pump_flat, nA_buf, nI_buf, psi_buf)

                # Record event so transfer stream can wait for compute
                self._transfer_event.record(self._stream_compute)

            else:
                psi_flat = self._cast_to_precision(state.psi).ravel()
                nR_flat = self._cast_to_precision(
                    reservoir.get_reservoir_density()
                ).ravel()
                psi_buf = self._psi_tmp
                psi_buf[:] = psi_flat
                nR_buf = self._nA_tmp
                nR_buf[:] = nR_flat

                self._step_gpu(V_flat, pump_flat, nR_buf, psi_buf)

                self._transfer_event.record(self._stream_compute)

        # Wait for compute, then write back results
        # (transfer stream waits on compute event)
        self._stream_transfer.wait_event(self._transfer_event)
        with self._stream_transfer:
            if self._reservoir_type == "double":
                # Cast back to original precision if needed
                psi_result = psi_buf.reshape(shape)
                if state.psi.dtype != psi_result.dtype:
                    psi_result = psi_result.astype(state.psi.dtype)
                state.psi = psi_result
                state.psi = boundary_condition.after_step_action(state.psi)

                nA_result = nA_buf.reshape(shape)
                nI_result = nI_buf.reshape(shape)
                if nA.dtype != nA_result.dtype:
                    nA_result = nA_result.astype(nA.dtype)
                    nI_result = nI_result.astype(nI.dtype)
                reservoir.set_state((
                    self.xp.maximum(nA_result, 0),
                    self.xp.maximum(nI_result, 0),
                ))
            else:
                psi_result = psi_buf.reshape(shape)
                if state.psi.dtype != psi_result.dtype:
                    psi_result = psi_result.astype(state.psi.dtype)
                state.psi = psi_result
                state.psi = boundary_condition.after_step_action(state.psi)

                nR_result = nR_buf.reshape(shape)
                if reservoir.get_reservoir_density().dtype != nR_result.dtype:
                    nR_result = nR_result.astype(
                        reservoir.get_reservoir_density().dtype
                    )
                reservoir.set_state((self.xp.maximum(nR_result, 0),))

        # Synchronize transfer stream to ensure state is ready
        self._stream_transfer.synchronize()
