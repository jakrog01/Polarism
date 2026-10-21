"""CUDA RK4 solver kernels and driver code."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from polarism.compute_engine import has_full_rate_fp64
from polarism.config.batch_physics import BatchPhysics
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.reservoir.abstract_reservoir import AbstractReservoir
    from polarism.grid.simulation_grid_2d import SimulationGrid2D

PREAMBLE_1D = """
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;
    const int i = tid / nx;
    const int j = tid % nx;
"""

_NEUMANN_NEIGHBORS = """
    const int up = (i > 0)    ? tid - nx : tid + nx;
    const int dn = (i < ny-1) ? tid + nx : tid - nx;
    const int lt = (j > 0)    ? tid - 1  : tid + 1;
    const int rt = (j < nx-1) ? tid + 1  : tid - 1;
"""

_PERIODIC_NEIGHBORS = """
    const int up = ((i > 0)    ? i - 1 : ny - 1) * nx + j;
    const int dn = ((i < ny-1) ? i + 1 : 0)      * nx + j;
    const int lt = i * nx + ((j > 0)    ? j - 1 : nx - 1);
    const int rt = i * nx + ((j < nx-1) ? j + 1 : 0);
"""

_KERNEL_RHS_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void gpe_rhs(
    const {REAL}* __restrict__ psi,
    const {REAL}* __restrict__ nR,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C, const {REAL} gamma_R,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_R
) {{
    {PREAMBLE}

    const {REAL} pre = psi[2*tid];
    const {REAL} pim = psi[2*tid + 1];

    {NEIGHBOR_CODE}
    {DIAG_NEIGHBORS}

    {LAP_BLOCK_RHS}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} nR_val = nR[tid];
    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nR_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    const {REAL} lap_nR = (nR[dn] - 2*nR_val + nR[up]) * inv_dy2
                        + (nR[rt] - 2*nR_val + nR[lt]) * inv_dx2;
    dnR_dt[tid] = pump_scale * pump[tid] - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;
}}
"""

_KERNEL_FUSED_STAGE_RHS_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void {KERNEL_NAME}(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k_nR,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ {PSI_OUTPUT},
    {REAL}* __restrict__ {NR_OUTPUT},
    {EXTRA_ARGUMENTS}
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} c_dt,
    {DT6_ARGUMENT}
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R,
    const {REAL} R_cond, const {REAL} gamma_C, const {REAL} gamma_R,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_R
) {{
    {PREAMBLE}

    const {REAL} pre = psi0[2*tid]   + c_dt * k_psi[2*tid];
    const {REAL} pim = psi0[2*tid+1] + c_dt * k_psi[2*tid+1];
    const {REAL} nR_val = nR0[tid] + c_dt * k_nR[tid];

    {NEIGHBOR_CODE}

        {DIAG_NEIGHBORS}

    const {REAL} psi_up_re = psi0[2*up]   + c_dt * k_psi[2*up];
    const {REAL} psi_up_im = psi0[2*up+1] + c_dt * k_psi[2*up+1];
    const {REAL} psi_dn_re = psi0[2*dn]   + c_dt * k_psi[2*dn];
    const {REAL} psi_dn_im = psi0[2*dn+1] + c_dt * k_psi[2*dn+1];
    const {REAL} psi_lt_re = psi0[2*lt]   + c_dt * k_psi[2*lt];
    const {REAL} psi_lt_im = psi0[2*lt+1] + c_dt * k_psi[2*lt+1];
    const {REAL} psi_rt_re = psi0[2*rt]   + c_dt * k_psi[2*rt];
    const {REAL} psi_rt_im = psi0[2*rt+1] + c_dt * k_psi[2*rt+1];
    {EXTRA_PSI_DIAG}

    {LAP_BLOCK_FUSED}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nR_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    {PSI_EPILOGUE}

    const {REAL} nR_up = nR0[up] + c_dt * k_nR[up];
    const {REAL} nR_dn = nR0[dn] + c_dt * k_nR[dn];
    const {REAL} nR_lt = nR0[lt] + c_dt * k_nR[lt];
    const {REAL} nR_rt = nR0[rt] + c_dt * k_nR[rt];
    const {REAL} lap_nR = (nR_dn - 2*nR_val + nR_up) * inv_dy2
                        + (nR_rt - 2*nR_val + nR_lt) * inv_dx2;
    {RESERVOIR_EPILOGUE}
}}
"""

_KERNEL_RHS_DOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void gpe_rhs_double(
    const {REAL}* __restrict__ psi,
    const {REAL}* __restrict__ nA,
    const {REAL}* __restrict__ nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnA_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R, const {REAL} g_I,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_I, const {REAL} gamma_A,
    const {REAL} R_IA, const {REAL} R_AI,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_I, const {REAL} diff_A
) {{
    {PREAMBLE}

    const {REAL} pre = psi[2*tid];
    const {REAL} pim = psi[2*tid + 1];

    {NEIGHBOR_CODE}
    {DIAG_NEIGHBORS}

    {LAP_BLOCK_RHS}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} nA_val = nA[tid];
    const {REAL} nI_val = nI[tid];

    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nA_val + g_I * nI_val;
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nA_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    const {REAL} lap_nI = (nI[dn] - 2*nI_val + nI[up]) * inv_dy2
                        + (nI[rt] - 2*nI_val + nI[lt]) * inv_dx2;
    const {REAL} lap_nA = (nA[dn] - 2*nA_val + nA[up]) * inv_dy2
                        + (nA[rt] - 2*nA_val + nA[lt]) * inv_dx2;
    dnI_dt[tid] = pump_scale * pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val + diff_I * lap_nI;
    dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val + diff_A * lap_nA;
}}
"""

_KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void {KERNEL_NAME}(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nA0,
    const {REAL}* __restrict__ k_nA,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k_nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ {PSI_OUTPUT},
    {REAL}* __restrict__ {NA_OUTPUT},
    {REAL}* __restrict__ {NI_OUTPUT},
    {EXTRA_ARGUMENTS}
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} c_dt,
    {DT6_ARGUMENT}
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R, const {REAL} g_I,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_I, const {REAL} gamma_A,
    const {REAL} R_IA, const {REAL} R_AI,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_I, const {REAL} diff_A
) {{
    {PREAMBLE}

    const {REAL} pre = psi0[2*tid]   + c_dt * k_psi[2*tid];
    const {REAL} pim = psi0[2*tid+1] + c_dt * k_psi[2*tid+1];
    const {REAL} nA_val = nA0[tid] + c_dt * k_nA[tid];
    const {REAL} nI_val = nI0[tid] + c_dt * k_nI[tid];

    {NEIGHBOR_CODE}
    {DIAG_NEIGHBORS}

    const {REAL} psi_up_re = psi0[2*up]   + c_dt * k_psi[2*up];
    const {REAL} psi_up_im = psi0[2*up+1] + c_dt * k_psi[2*up+1];
    const {REAL} psi_dn_re = psi0[2*dn]   + c_dt * k_psi[2*dn];
    const {REAL} psi_dn_im = psi0[2*dn+1] + c_dt * k_psi[2*dn+1];
    const {REAL} psi_lt_re = psi0[2*lt]   + c_dt * k_psi[2*lt];
    const {REAL} psi_lt_im = psi0[2*lt+1] + c_dt * k_psi[2*lt+1];
    const {REAL} psi_rt_re = psi0[2*rt]   + c_dt * k_psi[2*rt];
    const {REAL} psi_rt_im = psi0[2*rt+1] + c_dt * k_psi[2*rt+1];
    {EXTRA_PSI_DIAG}

    {LAP_BLOCK_FUSED}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nA_val + g_I * nI_val;
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nA_val * hbar_over_2m;

    {PSI_EPILOGUE}

    const {REAL} nI_up = nI0[up] + c_dt * k_nI[up];
    const {REAL} nI_dn = nI0[dn] + c_dt * k_nI[dn];
    const {REAL} nI_lt = nI0[lt] + c_dt * k_nI[lt];
    const {REAL} nI_rt = nI0[rt] + c_dt * k_nI[rt];
    const {REAL} lap_nI = (nI_dn - 2*nI_val + nI_up) * inv_dy2
                        + (nI_rt - 2*nI_val + nI_lt) * inv_dx2;

    const {REAL} nA_up = nA0[up] + c_dt * k_nA[up];
    const {REAL} nA_dn = nA0[dn] + c_dt * k_nA[dn];
    const {REAL} nA_lt = nA0[lt] + c_dt * k_nA[lt];
    const {REAL} nA_rt = nA0[rt] + c_dt * k_nA[rt];
    const {REAL} lap_nA = (nA_dn - 2*nA_val + nA_up) * inv_dy2
                        + (nA_rt - 2*nA_val + nA_lt) * inv_dx2;

    {RESERVOIR_EPILOGUE}
}}
"""

_KERNEL_RHS_QDOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void gpe_rhs_qdouble(
    const {REAL}* __restrict__ psi,
    const {REAL}* __restrict__ nR,
    const {REAL}* __restrict__ nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R, const {REAL} g_I,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_R, const {REAL} gamma_I, const {REAL} kappa,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_I, const {REAL} diff_R
) {{
    {PREAMBLE}

    const {REAL} pre = psi[2*tid];
    const {REAL} pim = psi[2*tid + 1];

    {NEIGHBOR_CODE}
    {DIAG_NEIGHBORS}

    {LAP_BLOCK_RHS}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} nR_val = nR[tid];
    const {REAL} nI_val = nI[tid];
    const {REAL} transfer = kappa * nI_val * nI_val;

    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nR_val + g_I * nI_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    const {REAL} lap_nR = (nR[dn] - 2*nR_val + nR[up]) * inv_dy2
                        + (nR[rt] - 2*nR_val + nR[lt]) * inv_dx2;
    const {REAL} lap_nI = (nI[dn] - 2*nI_val + nI[up]) * inv_dy2
                        + (nI[rt] - 2*nI_val + nI[lt]) * inv_dx2;
    dnR_dt[tid] = transfer - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;
    dnI_dt[tid] = pump_scale * pump[tid] - transfer - gamma_I * nI_val + diff_I * lap_nI;
}}
"""

_KERNEL_FUSED_STAGE_RHS_QDOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void {KERNEL_NAME}(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k_nR,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k_nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ {PSI_OUTPUT},
    {REAL}* __restrict__ {NR_OUTPUT},
    {REAL}* __restrict__ {NI_OUTPUT},
    {EXTRA_ARGUMENTS}
    const int nx, const int ny,
    const {REAL} pump_scale, const {REAL} v_im_scale,
    const {REAL} c_dt,
    {DT6_ARGUMENT}
    const {REAL} inv_dx2, const {REAL} inv_dy2,
    const {REAL} neg_hbar2_over_2m,
    const {REAL} inv_hbar,
    const {REAL} g_C, const {REAL} g_R, const {REAL} g_I,
    const {REAL} R_cond, const {REAL} gamma_C,
    const {REAL} gamma_R, const {REAL} gamma_I, const {REAL} kappa,
    const {REAL} kinetic_relaxation_eta, const {REAL} diff_I, const {REAL} diff_R
) {{
    {PREAMBLE}

    const {REAL} pre = psi0[2*tid]   + c_dt * k_psi[2*tid];
    const {REAL} pim = psi0[2*tid+1] + c_dt * k_psi[2*tid+1];
    const {REAL} nR_val = nR0[tid] + c_dt * k_nR[tid];
    const {REAL} nI_val = nI0[tid] + c_dt * k_nI[tid];

    {NEIGHBOR_CODE}
    {DIAG_NEIGHBORS}

    const {REAL} psi_up_re = psi0[2*up]   + c_dt * k_psi[2*up];
    const {REAL} psi_up_im = psi0[2*up+1] + c_dt * k_psi[2*up+1];
    const {REAL} psi_dn_re = psi0[2*dn]   + c_dt * k_psi[2*dn];
    const {REAL} psi_dn_im = psi0[2*dn+1] + c_dt * k_psi[2*dn+1];
    const {REAL} psi_lt_re = psi0[2*lt]   + c_dt * k_psi[2*lt];
    const {REAL} psi_lt_im = psi0[2*lt+1] + c_dt * k_psi[2*lt+1];
    const {REAL} psi_rt_re = psi0[2*rt]   + c_dt * k_psi[2*rt];
    const {REAL} psi_rt_im = psi0[2*rt+1] + c_dt * k_psi[2*rt+1];
    {EXTRA_PSI_DIAG}

    {LAP_BLOCK_FUSED}

    const {REAL} rho = pre*pre + pim*pim;
    const {REAL} transfer = kappa * nI_val * nI_val;

    const {REAL} eff = V_re[tid] + g_C * rho + g_R * nR_val + g_I * nI_val;
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + v_im_scale * V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    {PSI_EPILOGUE}

    const {REAL} nR_up = nR0[up] + c_dt * k_nR[up];
    const {REAL} nR_dn = nR0[dn] + c_dt * k_nR[dn];
    const {REAL} nR_lt = nR0[lt] + c_dt * k_nR[lt];
    const {REAL} nR_rt = nR0[rt] + c_dt * k_nR[rt];
    const {REAL} lap_nR = (nR_dn - 2*nR_val + nR_up) * inv_dy2
                        + (nR_rt - 2*nR_val + nR_lt) * inv_dx2;

    const {REAL} nI_up = nI0[up] + c_dt * k_nI[up];
    const {REAL} nI_dn = nI0[dn] + c_dt * k_nI[dn];
    const {REAL} nI_lt = nI0[lt] + c_dt * k_nI[lt];
    const {REAL} nI_rt = nI0[rt] + c_dt * k_nI[rt];
    const {REAL} lap_nI = (nI_dn - 2*nI_val + nI_up) * inv_dy2
                        + (nI_rt - 2*nI_val + nI_lt) * inv_dx2;

    {RESERVOIR_EPILOGUE}
}}
"""

def _build_laplacian_code(
    laplacian_type: str,
    bc_type: str,
    real_type: str,
) -> tuple[str, str, str, str]:
    """Return (diag_neighbors, extra_psi_diag, lap_block_rhs, lap_block_fused).

    The first value goes into the RHS and fused templates after the cardinal
    neighbors.  The second is appended to the fused cardinal-psi declarations.
    The third and fourth replace the lap_re/lap_im lines in RHS and fused kernels.
    """
    rt = real_type
    if laplacian_type == "five-point":
        lap_block_rhs = (
            f"const {rt} lap_re = (psi[2*dn]   - 2*pre + psi[2*up])   * inv_dy2\n"
            f"                        + (psi[2*rt]   - 2*pre + psi[2*lt])   * inv_dx2;\n"
            f"    const {rt} lap_im = (psi[2*dn+1] - 2*pim + psi[2*up+1]) * inv_dy2\n"
            f"                        + (psi[2*rt+1] - 2*pim + psi[2*lt+1]) * inv_dx2;"
        )
        lap_block_fused = (
            f"const {rt} lap_re = (psi_dn_re - 2*pre + psi_up_re) * inv_dy2\n"
            f"                        + (psi_rt_re - 2*pre + psi_lt_re) * inv_dx2;\n"
            f"    const {rt} lap_im = (psi_dn_im - 2*pim + psi_up_im) * inv_dy2\n"
            f"                        + (psi_rt_im - 2*pim + psi_lt_im) * inv_dx2;"
        )
        return "", "", lap_block_rhs, lap_block_fused

    if laplacian_type == "isotropic-9pt":
        if bc_type == "closed-interval":
            diag_neighbors = (
                "const int i_up = (i > 0)    ? i - 1 : 1;\n"
                "    const int i_dn = (i < ny-1) ? i + 1 : ny - 2;\n"
                "    const int j_lt = (j > 0)    ? j - 1 : 1;\n"
                "    const int j_rt = (j < nx-1) ? j + 1 : nx - 2;\n"
                "    const int ul = i_up * nx + j_lt;\n"
                "    const int ur = i_up * nx + j_rt;\n"
                "    const int dl = i_dn * nx + j_lt;\n"
                "    const int dr = i_dn * nx + j_rt;"
            )
        else:
            diag_neighbors = (
                "const int i_up = (i > 0)    ? i - 1 : ny - 1;\n"
                "    const int i_dn = (i < ny-1) ? i + 1 : 0;\n"
                "    const int j_lt = (j > 0)    ? j - 1 : nx - 1;\n"
                "    const int j_rt = (j < nx-1) ? j + 1 : 0;\n"
                "    const int ul = i_up * nx + j_lt;\n"
                "    const int ur = i_up * nx + j_rt;\n"
                "    const int dl = i_dn * nx + j_lt;\n"
                "    const int dr = i_dn * nx + j_rt;"
            )
        extra_psi_diag = (
            f"const {rt} psi_ul_re = psi0[2*ul]   + c_dt * k_psi[2*ul];\n"
            f"    const {rt} psi_ul_im = psi0[2*ul+1] + c_dt * k_psi[2*ul+1];\n"
            f"    const {rt} psi_ur_re = psi0[2*ur]   + c_dt * k_psi[2*ur];\n"
            f"    const {rt} psi_ur_im = psi0[2*ur+1] + c_dt * k_psi[2*ur+1];\n"
            f"    const {rt} psi_dl_re = psi0[2*dl]   + c_dt * k_psi[2*dl];\n"
            f"    const {rt} psi_dl_im = psi0[2*dl+1] + c_dt * k_psi[2*dl+1];\n"
            f"    const {rt} psi_dr_re = psi0[2*dr]   + c_dt * k_psi[2*dr];\n"
            f"    const {rt} psi_dr_im = psi0[2*dr+1] + c_dt * k_psi[2*dr+1];"
        )
        lap_block_rhs = (
            f"const {rt} inv_6h2 = inv_dx2 / ({rt})6.0;\n"
            f"    const {rt} lap_re = (({rt})4.0*(psi[2*dn]+psi[2*up]+psi[2*rt]+psi[2*lt])\n"
            f"                         + psi[2*ul]+psi[2*ur]+psi[2*dl]+psi[2*dr]\n"
            f"                         - ({rt})20.0*pre) * inv_6h2;\n"
            f"    const {rt} lap_im = (({rt})4.0*(psi[2*dn+1]+psi[2*up+1]+psi[2*rt+1]+psi[2*lt+1])\n"
            f"                         + psi[2*ul+1]+psi[2*ur+1]+psi[2*dl+1]+psi[2*dr+1]\n"
            f"                         - ({rt})20.0*pim) * inv_6h2;"
        )
        lap_block_fused = (
            f"const {rt} inv_6h2 = inv_dx2 / ({rt})6.0;\n"
            f"    const {rt} lap_re = (({rt})4.0*(psi_dn_re+psi_up_re+psi_rt_re+psi_lt_re)\n"
            f"                         + psi_ul_re+psi_ur_re+psi_dl_re+psi_dr_re\n"
            f"                         - ({rt})20.0*pre) * inv_6h2;\n"
            f"    const {rt} lap_im = (({rt})4.0*(psi_dn_im+psi_up_im+psi_rt_im+psi_lt_im)\n"
            f"                         + psi_ul_im+psi_ur_im+psi_dl_im+psi_dr_im\n"
            f"                         - ({rt})20.0*pim) * inv_6h2;"
        )
        return diag_neighbors, extra_psi_diag, lap_block_rhs, lap_block_fused

    raise ValueError(
        f"Unknown laplacian_type '{laplacian_type}'. "
        "Valid values: 'five-point', 'isotropic-9pt'."
    )


def build_kernel_source_1d(
    bc_type: str,
    real_type: str = "double",
    reservoir_type: str = "single",
    laplacian_type: str = "five-point",
    preamble: str = PREAMBLE_1D,
    combine_preamble: str = "",
    launch_bounds: str = "",
) -> tuple[str, str, str]:
    """Build the RHS, intermediate stage, and final stage CUDA sources."""
    neighbor_code = (
        _NEUMANN_NEIGHBORS if bc_type == "closed-interval" else _PERIODIC_NEIGHBORS
    )
    if reservoir_type == "double":
        rhs_tmpl = _KERNEL_RHS_DOUBLE_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE
        suffix = "_double"
        reservoir_fields = (
            ("a", "nA0", "k_nA", "dnA_dt", "nA_out",
             "R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val + diff_A * lap_nA"),
            ("i", "nI0", "k_nI", "dnI_dt", "nI_out",
             "pump_scale * pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val + diff_I * lap_nI"),
        )
    elif reservoir_type == "quadratic-double":
        rhs_tmpl = _KERNEL_RHS_QDOUBLE_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_QDOUBLE_TEMPLATE
        suffix = "_qdouble"
        reservoir_fields = (
            ("r", "nR0", "k_nR", "dnR_dt", "nR_out",
             "transfer - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR"),
            ("i", "nI0", "k_nI", "dnI_dt", "nI_out",
             "pump_scale * pump[tid] - transfer - gamma_I * nI_val + diff_I * lap_nI"),
        )
    else:
        rhs_tmpl = _KERNEL_RHS_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_TEMPLATE
        suffix = ""
        reservoir_fields = (
            ("n", "nR0", "k_nR", "dnR_dt", "nR_out",
             "pump_scale * pump[tid] - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR"),
        )

    diag_neighbors, extra_psi_diag, lap_block_rhs, lap_block_fused = _build_laplacian_code(
        laplacian_type, bc_type, real_type
    )
    fmt = dict(
        REAL=real_type,
        NEIGHBOR_CODE=neighbor_code,
        PREAMBLE=preamble,
        LAUNCH_BOUNDS=launch_bounds,
        DIAG_NEIGHBORS=diag_neighbors,
        EXTRA_PSI_DIAG=extra_psi_diag,
        LAP_BLOCK_RHS=lap_block_rhs,
        LAP_BLOCK_FUSED=lap_block_fused,
    )
    # Keep the former combine_preamble argument for positional callers. The
    # third returned source now computes stage 4 and writes the final state.
    _ = combine_preamble
    output_names = {"PSI_OUTPUT": "dpsi"}
    final_output_names = {"PSI_OUTPUT": "psi_out"}
    stage_reservoir = []
    final_reservoir = []
    extra_arguments = [
        f"const {real_type}* __restrict__ k1p,",
        f"const {real_type}* __restrict__ k2p,",
    ]
    for key, initial, derivative, stage_output, final_output, rhs in reservoir_fields:
        output_names[{"a": "NA_OUTPUT", "i": "NI_OUTPUT", "r": "NR_OUTPUT", "n": "NR_OUTPUT"}[key]] = stage_output
        final_output_names[{"a": "NA_OUTPUT", "i": "NI_OUTPUT", "r": "NR_OUTPUT", "n": "NR_OUTPUT"}[key]] = final_output
        stage_reservoir.append(f"{stage_output}[tid] = {rhs};")
        final_reservoir.extend((
            f"const {real_type} k4{key} = {rhs};",
            f"{real_type} {key}_new = {initial}[tid] + dt6 * (k1{key}[tid] + 2*k2{key}[tid] + 2*{derivative}[tid] + k4{key});",
            f"{final_output}[tid] = ({key}_new < 0) ? 0 : {key}_new;",
        ))
        extra_arguments.extend((
            f"const {real_type}* __restrict__ k1{key},",
            f"const {real_type}* __restrict__ k2{key},",
        ))

    stage_fmt = dict(fmt, KERNEL_NAME=f"fused_stage_rhs{suffix}",
                     EXTRA_ARGUMENTS="", DT6_ARGUMENT="",
                     PSI_EPILOGUE=(
                         "dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;\n"
                         "    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;"
                     ),
                     RESERVOIR_EPILOGUE="\n    ".join(stage_reservoir),
                     **output_names)
    final_fmt = dict(fmt, KERNEL_NAME=f"fused_final_stage{suffix}",
                     EXTRA_ARGUMENTS="\n    ".join(extra_arguments),
                     DT6_ARGUMENT=f"const {real_type} dt6,",
                     PSI_EPILOGUE=(
                         f"const {real_type} k4p_re = inv_hbar * A_im + gl * pre + relax * lap_re;\n"
                         f"    const {real_type} k4p_im = -inv_hbar * A_re + gl * pim + relax * lap_im;\n"
                         "    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k_psi[2*tid]   + k4p_re);\n"
                         "    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k_psi[2*tid+1] + k4p_im);"
                     ),
                     RESERVOIR_EPILOGUE="\n    ".join(final_reservoir),
                     **final_output_names)
    # Round the pump product on its own (no FMA contraction) so the scaled
    # envelope is bitwise equal to the pump array computed by the laser.
    mul = "__fmul_rn" if real_type == "float" else "__dmul_rn"
    sources = (rhs_tmpl.format(**fmt), fused_tmpl.format(**stage_fmt), fused_tmpl.format(**final_fmt))
    return tuple(s.replace("pump_scale * pump[tid]", f"{mul}(pump_scale, pump[tid])") for s in sources)


def _tile_index(index: int, size: int, bc_type: str) -> int:
    """Map a virtual halo coordinate to a grid coordinate."""
    if bc_type == "periodic":
        return index % size
    period = 2 * (size - 1)
    folded = index % period
    return min(folded, period - folded)


def _tile_halo_indices(nx: int, ny: int, tile_x: int, tile_y: int,
                       block_x: int, block_y: int, bc_type: str):
    """Emulate the grid indices loaded by a tile and its four-cell halo."""
    import numpy as np

    x = np.array([_tile_index(block_x * tile_x + i - 4, nx, bc_type)
                  for i in range(tile_x + 8)])
    y = np.array([_tile_index(block_y * tile_y + i - 4, ny, bc_type)
                  for i in range(tile_y + 8)])
    return y[:, None] * nx + x[None, :]


def _tile_shared_bytes(tile_x: int, tile_y: int, real_type: str,
                       reservoir_type: str) -> int:
    fields = 3 if reservoir_type == "single" else 4
    byte_width = 4 if real_type == "float" else 8
    return byte_width * fields * (2 * (tile_x + 8) * (tile_y + 8)
                                  + 3 * tile_x * tile_y)


def _choose_tile_shape(real_type: str, reservoir_type: str, block_limit: int,
                       sm_limit: int):
    candidates = ((32, 16), (32, 8), (16, 8), (8, 8))
    for tile in candidates:
        needed = _tile_shared_bytes(*tile, real_type, reservoir_type)
        if needed <= block_limit and 2 * needed <= sm_limit:
            return tile
    for tile in candidates:
        if _tile_shared_bytes(*tile, real_type, reservoir_type) <= block_limit:
            return tile
    return None


def _prefer_tiled(precision: str, major: int, minor: int) -> bool:
    return precision == "single" or has_full_rate_fp64(major, minor)


def build_tiled_kernel_source(bc_type: str, real_type: str, reservoir_type: str,
                              laplacian_type: str, tile_x: int, tile_y: int,
                              complex_potential: bool, graph: bool = False) -> str:
    """Build a whole-step kernel from the staged RHS expressions."""
    import re

    rhs_source = build_kernel_source_1d(
        bc_type, real_type, reservoir_type, laplacian_type
    )[0]
    fields = ("nR",) if reservoir_type == "single" else (
        ("nA", "nI") if reservoir_type == "double" else ("nR", "nI")
    )
    derivative_names = {"nR": "dnR_dt", "nA": "dnA_dt", "nI": "dnI_dt"}
    suffix = {"single": "", "double": "_double",
              "quadratic-double": "_qdouble"}[reservoir_type]
    header, body = rhs_source.split(") {", 1)
    body = body.rsplit("\n}", 1)[0]
    body = body.replace(PREAMBLE_1D, "\n")
    neighbors = _NEUMANN_NEIGHBORS if bc_type == "closed-interval" else _PERIODIC_NEIGHBORS
    tile_neighbors = ("""
    const int up = tid - pitch;
    const int dn = tid + pitch;
    const int lt = tid - 1;
    const int rt = tid + 1;
""" if bc_type == "periodic" else """
    const int up = (tiled_coord(vy - 1, ny) == (i > 0 ? i - 1 : 1))
                   ? tid - pitch : tid + pitch;
    const int dn = (tiled_coord(vy + 1, ny) == (i < ny-1 ? i + 1 : ny-2))
                   ? tid + pitch : tid - pitch;
    const int lt = (tiled_coord(vx - 1, nx) == (j > 0 ? j - 1 : 1))
                   ? tid - 1 : tid + 1;
    const int rt = (tiled_coord(vx + 1, nx) == (j < nx-1 ? j + 1 : nx-2))
                   ? tid + 1 : tid - 1;
""")
    body = body.replace(neighbors, tile_neighbors)
    diag, _, _, _ = _build_laplacian_code(laplacian_type, bc_type, real_type)
    if diag:
        body = body.replace(diag, """
    const int ul = up + (lt - tid);
    const int ur = up + (rt - tid);
    const int dl = dn + (lt - tid);
    const int dr = dn + (rt - tid);
""")
    body = body.replace("V_re[tid]", "V_re[gid]").replace("V_im[tid]", "V_im[gid]")
    body = body.replace("pump[tid]", "pump[gid]")
    if not complex_potential:
        body = body.replace("V_im[gid]", "V_re[gid]")
    body = body.replace("dpsi[2*tid+1]", "dpsi[1]")
    body = body.replace("dpsi[2*tid]", "dpsi[0]")
    for name in derivative_names.values():
        body = body.replace(f"{name}[tid]", f"{name}[0]")
    parameter_declarations = f"const {real_type} inv_dx2," + rhs_source.split(
        f"const {real_type} inv_dx2,", 1
    )[1].split(") {", 1)[0]
    parameter_names = re.findall(rf"const {real_type} (\w+)", parameter_declarations)
    rhs_header = re.sub(
        r'^extern "C" __global__\s*void \w+\(',
        '__device__ __forceinline__ void tiled_rhs(', header.strip(), count=1,
    )
    rhs_header = rhs_header.replace(
        "const int nx, const int ny,",
        "const int nx, const int ny, const int tid, const int gid, "
        "const int vx, const int vy, const int pitch,",
    )
    if bc_type == "periodic":
        coord_source = """__device__ __forceinline__ int tiled_coord(int index, int size) {
    int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}"""
    else:
        coord_source = """__device__ __forceinline__ int tiled_coord(int index, int size) {
    const int period = 2 * (size - 1);
    int folded = index % period;
    if (folded < 0) folded += period;
    return folded < size ? folded : period - folded;
}"""
    declarations = [f"const {real_type}* __restrict__ psi_initial"]
    declarations += [f"const {real_type}* __restrict__ {name}_initial" for name in fields]
    declarations += [f"const {real_type}* __restrict__ V_re",
                     f"const {real_type}* __restrict__ V_im"]
    declarations += [f"const {real_type}* __restrict__ pump{stage}" for stage in range(3)]
    declarations += [f"{real_type}* __restrict__ psi_out"]
    declarations += [f"{real_type}* __restrict__ {name}_out" for name in fields]
    declarations += ["const int nx", "const int ny"]
    if graph:
        declarations += [f"const {real_type}* pump_scales", "const int pump_index"]
    else:
        declarations += [f"const {real_type} scale{stage}" for stage in range(3)]
    declarations += [f"const {real_type} v_im_scale",
                     f"const {real_type} half_dt", f"const {real_type} dt",
                     f"const {real_type} dt6", parameter_declarations]
    scale_values = ([f"pump_scales[pump_index + {i}]" for i in range(3)] if graph
                    else [f"scale{i}" for i in range(3)])
    count = 2 + len(fields)
    area = f"(({tile_x}+8)*({tile_y}+8))"
    lines = [
        f"extern __shared__ {real_type} scratch[];",
        f"const int pitch = {tile_x}+8;",
        f"const int area = {area};",
        "const int lane = threadIdx.x;",
        f"const int base_x = blockIdx.x * {tile_x};",
        f"const int base_y = blockIdx.y * {tile_y};",
    ]
    for bank in (0, 1):
        offset = bank * count
        lines.append(f"{real_type}* psi{bank} = scratch + {offset}*area;")
        for index, name in enumerate(fields):
            lines.append(f"{real_type}* {name}{bank} = scratch + {offset+2+index}*area;")
    k_base = 2 * count
    lines.append(f"{real_type}* kp = scratch + {k_base}*area;")
    for index, name in enumerate(fields):
        lines.append(f"{real_type}* k{name} = scratch + {k_base}*area + "
                     f"(2+{index})*{tile_x}*{tile_y}*3;")
    lines += [
        "for (int p = lane; p < area; p += blockDim.x) {",
        "    const int vx = base_x + p % pitch - 4;",
        "    const int vy = base_y + p / pitch - 4;",
        "    const int gid = tiled_coord(vy, ny)*nx + tiled_coord(vx, nx);",
        "    psi0[2*p] = psi_initial[2*gid];",
    ]
    lines.append("    psi0[2*p+1] = psi_initial[2*gid+1];")
    for name in fields:
        lines.append(f"    {name}0[p] = {name}_initial[gid];")
    lines += ["}", "__syncthreads();", "#pragma unroll",
              "for (int stage = 0; stage < 4; ++stage) {",
              "    for (int p = lane; p < area; p += blockDim.x) {",
              "        const int x = p % pitch;", "        const int y = p / pitch;",
              "        if (x < stage+1 || x >= pitch-stage-1 ||",
              f"            y < stage+1 || y >= {tile_y}+8-stage-1) continue;",
              "        const int vx = base_x + x - 4;",
              "        const int vy = base_y + y - 4;",
              "        const int gid = tiled_coord(vy, ny)*nx + tiled_coord(vx, nx);",
              "        const int out = (y-4)*" + str(tile_x) + " + x-4;",
              f"        {real_type} dp[2];"]
    for name in fields:
        lines.append(f"        {real_type} d{name}[1];")
    lines += [
        f"        const {real_type}* input_psi = (stage & 1) ? psi1 : psi0;",
    ]
    for name in fields:
        lines.append(f"        const {real_type}* input_{name} = (stage & 1) ? {name}1 : {name}0;")
    lines += [
        f"        const {real_type}* stage_pump = stage == 0 ? pump0 : "
        "(stage == 3 ? pump2 : pump1);",
        f"        const {real_type} stage_scale = stage == 0 ? {scale_values[0]} : "
        f"(stage == 3 ? {scale_values[2]} : {scale_values[1]});",
        "        tiled_rhs(input_psi, " + ", ".join(f"input_{name}" for name in fields)
        + ", V_re, V_im, stage_pump, dp, "
        + ", ".join(f"d{name}" for name in fields)
        + ", nx, ny, p, gid, vx, vy, pitch, stage_scale, v_im_scale, "
        + ", ".join(parameter_names) + ");",
        "        if (stage < 3) {",
        f"            const {real_type} c_dt = stage == 2 ? dt : half_dt;",
        f"            {real_type}* next_psi = (stage & 1) ? psi0 : psi1;",
        "            next_psi[2*p] = psi_initial[2*gid] + c_dt * dp[0];",
        "            next_psi[2*p+1] = psi_initial[2*gid+1] + c_dt * dp[1];",
    ]
    for name in fields:
        lines += [f"            {real_type}* next_{name} = (stage & 1) ? {name}0 : {name}1;",
                  f"            next_{name}[p] = {name}_initial[gid] + c_dt * d{name}[0];"]
    lines += ["        }", f"        if (x >= 4 && x < 4+{tile_x} && y >= 4 && y < 4+{tile_y}) {{",
              "            if (stage < 3) {",
              "                kp[stage*2*" + str(tile_x*tile_y) + "+2*out] = dp[0];",
              "                kp[stage*2*" + str(tile_x*tile_y) + "+2*out+1] = dp[1];"]
    for name in fields:
        lines.append(f"                k{name}[stage*{tile_x*tile_y}+out] = d{name}[0];")
    lines += ["            } else if (vx < nx && vy < ny) {",
              f"                psi_out[2*gid] = psi_initial[2*gid] + dt6 * (kp[2*out] + 2*kp[2*{tile_x*tile_y}+2*out] + 2*kp[4*{tile_x*tile_y}+2*out] + dp[0]);",
              f"                psi_out[2*gid+1] = psi_initial[2*gid+1] + dt6 * (kp[2*out+1] + 2*kp[2*{tile_x*tile_y}+2*out+1] + 2*kp[4*{tile_x*tile_y}+2*out+1] + dp[1]);"]
    for name in fields:
        lines += [f"                {real_type} {name}_new = {name}_initial[gid] + dt6 * (k{name}[out] + 2*k{name}[{tile_x*tile_y}+out] + 2*k{name}[2*{tile_x*tile_y}+out] + d{name}[0]);",
                  f"                {name}_out[gid] = ({name}_new < 0) ? 0 : {name}_new;"]
    lines += ["            }", "        }", "    }", "    __syncthreads();", "}"]
    kernel = "\n    ".join(lines)
    return (coord_source + "\n"
            + rhs_header + ") {\n    const int i = gid / nx;\n    const int j = gid % nx;\n"
            + body + "\n}\n"
            + f'extern "C" __global__ void tiled_step{suffix}(' + ",\n    ".join(declarations)
            + ") {\n    " + kernel + "\n}\n")


def _auto_tune_block_size(kern_rhs, _N: int) -> int:
    """Pick a CUDA block size for the kernel."""
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
    """Generic fused RK4 CUDA solver with overridable kernel and launch hooks."""

    def __init__(self, config: Config, grid: SimulationGrid2D, batch_physics=None):
        """Set up the rk 4 cuda solver."""
        super().__init__(config)
        self.batch_physics = BatchPhysics(config.physics, batch_physics, self.xp, getattr(config.solver, "precision", "double"))
        self.nx = grid.nx
        self.ny = grid.ny
        self.N = grid.nx * grid.ny
        self.grid_type = grid.boundary
        _res_cfg = getattr(config, "reservoir", None)
        self._reservoir_type = (
            getattr(_res_cfg, "reservoir_type", "single") if _res_cfg else "single"
        )
        if self._reservoir_type not in ("single", "double", "quadratic-double"):
            raise ValueError(
                f"rk4-cuda has hard-coded reservoir kernels and does not support "
                f"reservoir_type='{self._reservoir_type}'."
            )

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

        self._hbar = self.batch_physics.get("hbar")
        self._m_eff = self.batch_physics.get("m_eff")
        self._g_C = self.batch_physics.get("g_C")
        self._g_R = self.batch_physics.get("g_R")
        self._g_I = self.batch_physics.get("g_I")
        self._R = self.batch_physics.get("R")
        self._gamma_C = self.batch_physics.get("gamma_C")
        self._gamma_R = self.batch_physics.get("gamma_R")
        self._dt = config.solver.dt
        self._kinetic_relaxation_eta = self.batch_physics.get("kinetic_relaxation_eta")
        self._reservoir_diffusion_I = self.batch_physics.get("reservoir_diffusion_I")
        self._reservoir_diffusion_A = self.batch_physics.get("reservoir_diffusion_A")
        self._reservoir_diffusion_R = self.batch_physics.get("reservoir_diffusion_R")

        if self._reservoir_type == "double":
            self._gamma_I = self.batch_physics.get("gamma_I")
            self._gamma_A = self.batch_physics.get("gamma_A")
            self._R_IA = self.batch_physics.get("R_IA")
            self._R_AI = self.batch_physics.get("R_AI")
        elif self._reservoir_type == "quadratic-double":
            self._gamma_I = self.batch_physics.get("gamma_I")
            self._kappa = self.batch_physics.get("kappa")

        self._inv_dx2 = 1.0 / (grid.dx ** 2)
        self._inv_dy2 = 1.0 / (grid.dy ** 2)
        self._neg_hbar2_over_2m = -(self._hbar ** 2) / (2.0 * self._m_eff)
        self._inv_hbar = 1.0 / self._hbar

        laplacian = getattr(config.solver, "laplacian", "five-point")
        if laplacian not in ("five-point", "isotropic-9pt"):
            raise ValueError(
                f"solver.laplacian='{laplacian}' is not recognised. "
                "Valid values: 'five-point', 'isotropic-9pt'."
            )
        if laplacian == "isotropic-9pt":
            rel_diff = abs(grid.dx - grid.dy) / max(grid.dx, grid.dy)
            if rel_diff > 1e-8:
                raise ValueError(
                    f"solver.laplacian='isotropic-9pt' requires dx == dy, "
                    f"got dx={grid.dx}, dy={grid.dy} (relative difference {rel_diff:.2e})."
                )
        self._laplacian_type = laplacian

        self._use_gpu = hasattr(self.xp, "cuda") and self.batch_physics.size is None

        if self._use_gpu:
            _cast = self._np_real

            self._k_nx = _np.int32(self.nx)
            self._k_ny = _np.int32(self.ny)
            self._k_N = _np.int32(self.N)
            self._k_int = _np.int32
            self._k_inv_dx2 = _cast(self._inv_dx2)
            self._k_inv_dy2 = _cast(self._inv_dy2)
            self._k_neg_hbar2_over_2m = _cast(self._neg_hbar2_over_2m)
            self._k_inv_hbar = _cast(self._inv_hbar)
            self._k_g_C = _cast(self._g_C)
            self._k_g_R = _cast(self._g_R)
            self._k_g_I = _cast(self._g_I)
            self._k_R = _cast(self._R)
            self._k_gamma_C = _cast(self._gamma_C)
            self._k_gamma_R = _cast(self._gamma_R)

            self._k_half_dt = _cast(0.5 * self._dt)
            self._k_dt = _cast(self._dt)
            self._k_dt6 = _cast(self._dt / 6.0)
            self._k_kinetic_relaxation_eta = _cast(self._kinetic_relaxation_eta)
            self._k_diff_I = _cast(self._reservoir_diffusion_I)
            self._k_diff_A = _cast(self._reservoir_diffusion_A)
            self._k_diff_R = _cast(self._reservoir_diffusion_R)

            if self._reservoir_type == "double":
                self._k_gamma_I = _cast(self._gamma_I)
                self._k_gamma_A = _cast(self._gamma_A)
                self._k_R_IA = _cast(self._R_IA)
                self._k_R_AI = _cast(self._R_AI)
            elif self._reservoir_type == "quadratic-double":
                self._k_gamma_I = _cast(self._gamma_I)
                self._k_kappa = _cast(self._kappa)

            self._compile_kernels()
            self._init_gpu_buffers()

            common = (
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
            )
            if self._reservoir_type == "single":
                model = (
                    self._k_R, self._k_gamma_C, self._k_gamma_R,
                    self._k_kinetic_relaxation_eta, self._k_diff_R,
                )
            elif self._reservoir_type == "double":
                model = (
                    self._k_g_I, self._k_R, self._k_gamma_C,
                    self._k_gamma_I, self._k_gamma_A,
                    self._k_R_IA, self._k_R_AI,
                    self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_A,
                )
            else:
                model = (
                    self._k_g_I, self._k_R, self._k_gamma_C,
                    self._k_gamma_R, self._k_gamma_I, self._k_kappa,
                    self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_R,
                )
            self._k_rhs_parameters = common + model
            props = self.xp.cuda.runtime.getDeviceProperties(
                self.xp.cuda.runtime.getDevice()
            )
            self._use_tiled = _prefer_tiled(
                self._precision, props["major"], props["minor"]
            )
            self._tiled_kernels = {}
            self._tile_shape = self._select_tile_shape()
            self._graph_cache = {}
            self._graph_signature = None
            self._graphs_failed = False

    def _build_kernel_source(
        self, bc_type: str, real_type: str, reservoir_type: str
    ) -> tuple[str, str, str]:
        """Build the CUDA kernel source."""
        return build_kernel_source_1d(
            bc_type, real_type, reservoir_type,
            laplacian_type=self._laplacian_type,
        )

    def _select_block_size(self) -> int:
        """Pick the CUDA block size."""
        return _auto_tune_block_size(self._kern_rhs, self.N)

    def _make_launch_dims(self, block_size: int) -> None:
        """Set the CUDA launch dimensions."""
        grid = (self.N + block_size - 1) // block_size
        self._k_grid = (grid,)
        self._k_block = (block_size,)

    def _compile_kernels(self) -> None:
        """Compile the CUDA kernels."""
        import cupy as cp

        rhs_src, fused_src, final_src = self._build_kernel_source(
            self.grid_type, self._real_type, self._reservoir_type
        )

        if self._reservoir_type == "double":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_double")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_double")
            self._kern_final = cp.RawKernel(final_src, "fused_final_stage_double")
        elif self._reservoir_type == "quadratic-double":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_qdouble")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_qdouble")
            self._kern_final = cp.RawKernel(final_src, "fused_final_stage_qdouble")
        else:
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs")
            self._kern_final = cp.RawKernel(final_src, "fused_final_stage")

        block = self._select_block_size()
        self._make_launch_dims(block)

    def _select_tile_shape(self):
        """Choose a tile that fits device shared memory, favoring two resident blocks."""
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        block_limit = props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"])
        sm_limit = props.get("sharedMemPerMultiprocessor", block_limit)
        return _choose_tile_shape(self._real_type, self._reservoir_type,
                                  block_limit, sm_limit)

    def _tiled_kernel(self, complex_potential, graph=False):
        key = (complex_potential, graph)
        if key not in self._tiled_kernels:
            import cupy as cp

            tile_x, tile_y = self._tile_shape
            source = build_tiled_kernel_source(
                self.grid_type, self._real_type, self._reservoir_type,
                self._laplacian_type, tile_x, tile_y, complex_potential, graph,
            )
            suffix = {"single": "", "double": "_double",
                      "quadratic-double": "_qdouble"}[self._reservoir_type]
            kernel = cp.RawKernel(source, "tiled_step" + suffix)
            shared_bytes = _tile_shared_bytes(tile_x, tile_y, self._real_type,
                                               self._reservoir_type)
            if shared_bytes > cp.cuda.runtime.getDeviceProperties(
                    cp.cuda.runtime.getDevice())["sharedMemPerBlock"]:
                kernel.max_dynamic_shared_size_bytes = shared_bytes
            kernel.compile()
            self._tiled_kernels[key] = (kernel, shared_bytes)
        return self._tiled_kernels[key]

    def _init_gpu_buffers(self) -> None:
        """Allocate three RK4 derivative stages and constant potential buffer."""
        xp = self.xp
        N = self.N

        self._complex_dtype = xp.complex64 if self._precision == "single" else xp.complex128
        self._real_dtype = xp.float32 if self._precision == "single" else xp.float64
        n_fields = 1 if self._reservoir_type == "single" else 2
        self._k_psi = tuple(xp.empty(N, dtype=self._complex_dtype) for _ in range(3))
        self._k_reservoir = tuple(
            tuple(xp.empty(N, dtype=self._real_dtype) for _ in range(n_fields))
            for _ in range(3)
        )

        self._potential_cached = None
        self._V_re_flat = None
        self._V_im_flat = None

    def _cast_to_precision(self, arr):
        """Cast array to solver precision if needed (no-op when types match)."""
        xp = self.xp
        if self._precision == "single":
            if xp.iscomplexobj(arr):
                if arr.dtype != xp.complex64:
                    return arr.astype(xp.complex64)
            else:
                if arr.dtype != xp.float32:
                    return arr.astype(xp.float32)
        return arr

    def _prepare_potential_buffers(self, potential):
        """Return contiguous real/imag flattened potential buffers, cached by identity."""
        xp = self.xp
        if potential is self._potential_cached:
            return self._V_re_flat, self._V_im_flat

        if xp.iscomplexobj(potential):
            V_re = self._cast_to_precision(xp.ascontiguousarray(xp.real(potential)))
            V_im = self._cast_to_precision(xp.ascontiguousarray(xp.imag(potential)))
            self._V_re_flat = V_re.ravel()
            self._V_im_flat = V_im.ravel()
        else:
            V_re = potential if potential.flags["C_CONTIGUOUS"] else xp.ascontiguousarray(potential)
            self._V_re_flat = self._cast_to_precision(V_re).ravel()
            self._V_im_flat = self._V_re_flat

        self._potential_cached = potential
        return self._V_re_flat, self._V_im_flat

    def _flat_gpu_input(self, array, dtype):
        """Return a contiguous flat input, preserving an already matching array."""
        if array.dtype != dtype:
            array = array.astype(dtype)
        return self.xp.ascontiguousarray(array).ravel()

    def _step_gpu(self, psi0, reservoir0, V_re, V_im, pumps, pump_scales,
                  v_im_scale, psi_out, reservoir_out, scale_buffer=None,
                  scale_offset=0):
        """Launch the same RK4 kernels for each supported reservoir model."""
        if self._use_tiled and self._tile_shape is not None:
            kernel, shared_bytes = self._tiled_kernel(bool(v_im_scale),
                                                       scale_buffer is not None)
            tile_x, tile_y = self._tile_shape
            scale_args = ((scale_buffer, self._k_int(scale_offset))
                          if scale_buffer is not None else pump_scales)
            kernel(
                ((self.nx + tile_x - 1) // tile_x,
                 (self.ny + tile_y - 1) // tile_y), (256,),
                (psi0, *reservoir0, V_re, V_im, *pumps,
                 psi_out, *reservoir_out, self._k_nx, self._k_ny,
                 *scale_args, v_im_scale, self._k_half_dt, self._k_dt,
                 self._k_dt6, *self._k_rhs_parameters),
                shared_mem=shared_bytes,
            )
            return
        grid, block = self._k_grid, self._k_block
        dimensions = (self._k_nx, self._k_ny)
        parameters = self._k_rhs_parameters
        k_psi = self._k_psi
        k_reservoir = self._k_reservoir
        rhs, fused, final = (
            self._graph_kernels if scale_buffer is not None else
            (self._kern_rhs, self._kern_fused, self._kern_final)
        )

        def scale_args(stage):
            if scale_buffer is None:
                return (pump_scales[stage],)
            return (scale_buffer, self._k_int(scale_offset + stage))

        rhs(
            grid, block,
            (psi0, *reservoir0, V_re, V_im, pumps[0],
             k_psi[0], *k_reservoir[0], *dimensions,
             *scale_args(0), v_im_scale, *parameters),
        )
        for stage in (1, 2):
            staged_reservoir = tuple(
                value for field, derivative in zip(reservoir0, k_reservoir[stage - 1])
                for value in (field, derivative)
            )
            fused(
                grid, block,
                (psi0, k_psi[stage - 1], *staged_reservoir,
                 V_re, V_im, pumps[1], k_psi[stage], *k_reservoir[stage],
                 *dimensions, *scale_args(1), v_im_scale,
                 self._k_half_dt, *parameters),
            )

        staged_reservoir = tuple(
            value for field, derivative in zip(reservoir0, k_reservoir[2])
            for value in (field, derivative)
        )
        earlier_reservoir = tuple(
            value for derivatives in zip(k_reservoir[0], k_reservoir[1])
            for value in derivatives
        )
        final(
            grid, block,
            (psi0, k_psi[2], *staged_reservoir,
             V_re, V_im, pumps[2], psi_out, *reservoir_out,
             k_psi[0], k_psi[1], *earlier_reservoir,
             *dimensions, *scale_args(2), v_im_scale,
             self._k_dt, self._k_dt6, *parameters),
        )

    def _compile_graph_kernels(self):
        """Compile variants whose pump scale is read from a device array."""
        import re

        if hasattr(self, "_graph_kernels"):
            return
        sources = self._build_kernel_source(
            self.grid_type, self._real_type, self._reservoir_type
        )
        names = ("gpe_rhs", "fused_stage_rhs", "fused_final_stage")
        suffix = {"single": "", "double": "_double",
                  "quadratic-double": "_qdouble"}[self._reservoir_type]
        kernels = []
        for source, name in zip(sources, names):
            source = source.replace(
                f"const {self._real_type} pump_scale,",
                f"const {self._real_type}* pump_scales, const int pump_index,",
            )
            source = re.sub(r"\bpump_scale\b", "pump_scales[pump_index]", source)
            kernel = self.xp.RawKernel(source, name + suffix)
            kernel.compile()
            kernels.append(kernel)
        self._graph_kernels = tuple(kernels)

    def _ensure_advance_buffers(self):
        if hasattr(self, "_advance_buffers"):
            return
        xp = self.xp
        fields = 1 if self._reservoir_type == "single" else 2
        psi = tuple(xp.empty(self.N, self._complex_dtype) for _ in range(2))
        res = tuple(tuple(xp.empty(self.N, self._real_dtype) for _ in range(fields))
                    for _ in range(2))
        self._advance_buffers = (psi, res)
        self._graph_scales = xp.empty(64 * 3, self._real_dtype)
        self._advance_stream = xp.cuda.Stream(non_blocking=True)

    def _graph_for(self, length, potential, envelope, pump_flat, v_im_scale,
                   combiner=None):
        signature = (id(potential), id(envelope),
                     combiner.signature if combiner is not None else int(pump_flat.data.ptr))
        if signature != self._graph_signature:
            self._graph_cache.clear()
            self._graph_signature = signature
        if length in self._graph_cache:
            return self._graph_cache[length][0]
        if self._use_tiled and self._tile_shape is not None:
            self._tiled_kernel(bool(v_im_scale), graph=True)
        else:
            self._compile_graph_kernels()
        psi, res = self._advance_buffers
        scales = self._graph_scales
        vr, vi = self._prepare_potential_buffers(potential)
        stream = self._advance_stream
        stream.synchronize()
        with stream:
            stream.begin_capture()
            try:
                if combiner is not None:
                    start = combiner.combine(0, 0)
                for k in range(length):
                    if combiner is not None:
                        half = combiner.combine(2 * k + 1, 2)
                        end_index = 1 - k % 2
                        end = combiner.combine(2 * k + 2, end_index)
                        pumps = (start, half, end)
                        start = end
                    else:
                        pumps = (pump_flat,) * 3
                    self._step_gpu(
                        psi[k % 2], res[k % 2], vr, vi,
                        pumps, None, v_im_scale,
                        psi[(k + 1) % 2], res[(k + 1) % 2],
                        scales, 3 * k,
                    )
                graph = stream.end_capture()
            except Exception:
                try:
                    stream.end_capture()
                except Exception:
                    pass
                raise
        if len(self._graph_cache) == 4:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[length] = (graph, combiner if combiner is not None else pump_flat)
        return graph

    def advance(self, potential, pump, reservoir, boundary_condition, state,
                start_step: int, n_steps: int) -> None:
        """Run reusable CUDA chunks, or CuPy stages for batched physics.

        The tiled and graph kernels accept one scalar physics set. A physics
        batch uses the shape-aware CuPy RK4 stages until those kernels have a
        member-indexed parameter ABI.
        """
        if n_steps <= 0:
            return
        self.batch_physics.check(state.psi)
        if not self._use_gpu or state.psi.ndim == 3:
            return super().advance(potential, pump, reservoir, boundary_condition,
                                   state, start_step, n_steps)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir

        supported = {"single": SingleReservoir, "double": DoubleReservoir,
                     "quadratic-double": QuadraticDoubleReservoir}
        if (type(reservoir) is not supported[self._reservoir_type]
                or not getattr(boundary_condition.absorption, "after_step_is_noop", False)):
            return super().advance(potential, pump, reservoir, boundary_condition,
                                   state, start_step, n_steps)

        xp = self.xp
        shape = state.psi.shape
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        term = (terms[0] if terms is not None and len(terms) == 1 else
                pump.separable_term() if terms is None and hasattr(pump, "separable_term") else None)
        multi = terms is not None and len(terms) > 1
        no_lasers = not getattr(pump, "lasers", [True])
        if term is not None:
            envelope, coefficient = term
            pump_flat = self._flat_gpu_input(envelope, self._real_dtype)
        elif no_lasers:
            if not hasattr(self, "_zero_pump"):
                self._zero_pump = xp.zeros(shape, dtype=self._real_dtype)
            envelope = self._zero_pump
            pump_flat = self._flat_gpu_input(envelope, self._real_dtype)
            coefficient = lambda t: 1
        else:
            envelope = None
            coefficient = None
            pump_flat = None
        graph_ok = (
            self.config.solver.cuda_graphs and not self._graphs_failed
            and (coefficient is not None or multi)
            and getattr(boundary_condition.absorption, "after_step_is_noop", False)
        )
        vr, vi = self._prepare_potential_buffers(potential)
        vi_scale = self._np_real(1 if xp.iscomplexobj(potential) else 0)
        current_psi = self._flat_gpu_input(state.psi, self._complex_dtype)
        current_res = tuple(self._flat_gpu_input(x, self._real_dtype)
                            for x in reservoir.get_state())
        xp.cuda.get_current_stream().synchronize()
        self._ensure_advance_buffers()
        if multi:
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if not hasattr(self, "_pump_combiner"):
                self._pump_combiner = CudaPumpCombiner(xp, self._real_dtype, self.N)
            combiner = self._pump_combiner
            combiner.prepare(terms, 3, 2 * 64 + 1)
            xp.cuda.get_current_stream().synchronize()
        else:
            combiner = None
        work_psi, work_res = self._advance_buffers
        stream = self._advance_stream
        done = 0
        with stream:
            xp.copyto(work_psi[0], current_psi)
            for target, source in zip(work_res[0], current_res):
                xp.copyto(target, source)
            pending_hosts = []
            if graph_ok:
                try:
                    while n_steps - done >= 2:
                        remaining = n_steps - done
                        length = min(64, remaining - remaining % 2)
                        graph = self._graph_for(
                            length, potential, envelope, pump_flat, vi_scale, combiner
                        )
                        times = [time for k in range(length) for time in (
                            (start_step + done + k) * self._dt,
                            (start_step + done + k) * self._dt + 0.5 * self._dt,
                            (start_step + done + k) * self._dt + self._dt)]
                        if multi:
                            pump_times = [times[0]]
                            pump_times.extend(time for i, time in enumerate(times[1:])
                                              if i % 3 != 2)
                            pending_hosts.append(combiner.coefficients(
                                terms, pump_times, stream))
                            host_scales = self._np_real([1] * (length * 3))
                        else:
                            host_scales = self._np_real([coefficient(t) for t in times])
                        pending_hosts.append(host_scales)
                        self._graph_scales[:length * 3].set(host_scales, stream=stream)
                        graph.launch(stream=stream)
                        done += length
                        if len(pending_hosts) >= 32:
                            stream.synchronize()
                            pending_hosts.clear()
                except Exception as exc:
                    import sys
                    self._graphs_failed = True
                    self._graph_cache.clear()
                    self._graph_signature = None
                    print(f"rk4-cuda: CUDA graphs unavailable ({exc}); using regular launches.",
                          file=sys.stderr)
                    stream.synchronize()
                    xp.copyto(work_psi[0], current_psi)
                    for target, source in zip(work_res[0], current_res):
                        xp.copyto(target, source)
                    done = 0
            for k in range(done, n_steps):
                t = (start_step + k) * self._dt
                times = (t, t + 0.5 * self._dt, t + self._dt)
                if multi:
                    first = k == done
                    sample_times = times if first else times[1:]
                    pending_hosts.append(combiner.coefficients(terms, sample_times, stream))
                    if first:
                        start = combiner.combine(0, 0)
                    half = combiner.combine(1 if first else 0, 2)
                    end = combiner.combine(2 if first else 1, 1 - (k - done) % 2)
                    pumps = (start, half, end)
                    start = end
                    factors = (self._np_real(1),) * 3
                elif coefficient is None:
                    pumps = tuple(self._flat_gpu_input(pump(time), self._real_dtype)
                                  for time in times)
                    factors = (self._np_real(1),) * 3
                else:
                    pumps = (pump_flat,) * 3
                    factors = tuple(self._np_real(coefficient(time)) for time in times)
                parity = (k - done) % 2
                self._step_gpu(work_psi[parity], work_res[parity], vr, vi,
                               pumps, factors, vi_scale,
                               work_psi[1 - parity], work_res[1 - parity])
                if len(pending_hosts) >= 32:
                    stream.synchronize()
                    pending_hosts.clear()
            final_index = (n_steps - done) % 2
            state.psi = boundary_condition.after_step_action(
                work_psi[final_index].reshape(shape).copy()
            )
            reservoir.set_state(tuple(field.reshape(shape).copy()
                                      for field in work_res[final_index]))
            stream.synchronize()
        state.t = (start_step + n_steps) * self._dt

    def _cpu_laplacian_9pt(self, psi, out) -> None:
        """Apply the 9-point isotropic CPU Laplacian (requires dx == dy)."""
        xp = self.xp
        inv_6h2 = self._inv_dx2 / 6.0
        if self.grid_type == "periodic":
            up = xp.roll(psi, 1, axis=0)
            dn = xp.roll(psi, -1, axis=0)
            lt = xp.roll(psi, 1, axis=1)
            rt = xp.roll(psi, -1, axis=1)
            ul = xp.roll(up, 1, axis=1)
            ur = xp.roll(up, -1, axis=1)
            dl = xp.roll(dn, 1, axis=1)
            dr = xp.roll(dn, -1, axis=1)
            out[:] = (4 * (up + dn + lt + rt) + ul + ur + dl + dr - 20 * psi) * inv_6h2
            return
        padded = xp.pad(psi, 1, mode="reflect")
        N = padded[:-2, 1:-1]
        S = padded[2:, 1:-1]
        W = padded[1:-1, :-2]
        E = padded[1:-1, 2:]
        NW = padded[:-2, :-2]
        NE = padded[:-2, 2:]
        SW = padded[2:, :-2]
        SE = padded[2:, 2:]
        out[:] = (4 * (N + S + W + E) + NW + NE + SW + SE - 20 * psi) * inv_6h2

    def _cpu_laplacian(self, psi, out):
        """Apply the CPU Laplacian stencil."""
        if self._laplacian_type == "isotropic-9pt":
            if psi.ndim == 3:
                for member in range(psi.shape[0]):
                    self._cpu_laplacian_9pt(psi[member], out[member])
            else:
                self._cpu_laplacian_9pt(psi, out)
            return
        xp = self.xp
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        if self.grid_type == "periodic":
            out[..., :] = (
                xp.roll(psi, -1, axis=-2) - 2 * psi + xp.roll(psi, 1, axis=-2)
            ) * inv_dy2 + (
                xp.roll(psi, -1, axis=-1) - 2 * psi + xp.roll(psi, 1, axis=-1)
            ) * inv_dx2
            return

        ny, nx = self.ny, self.nx
        out.fill(0)
        if ny > 2 and nx > 2:
            out[..., 1:-1, 1:-1] = (
                psi[..., 2:, 1:-1] - 2 * psi[..., 1:-1, 1:-1] + psi[..., :-2, 1:-1]
            ) * inv_dy2 + (
                psi[..., 1:-1, 2:] - 2 * psi[..., 1:-1, 1:-1] + psi[..., 1:-1, :-2]
            ) * inv_dx2
        if nx > 2:
            out[..., 0, 1:-1] = (
                2 * (psi[..., 1, 1:-1] - psi[..., 0, 1:-1]) * inv_dy2
                + (psi[..., 0, 2:] - 2 * psi[..., 0, 1:-1] + psi[..., 0, :-2]) * inv_dx2
            )
            out[..., -1, 1:-1] = (
                2 * (psi[..., -2, 1:-1] - psi[..., -1, 1:-1]) * inv_dy2
                + (psi[..., -1, 2:] - 2 * psi[..., -1, 1:-1] + psi[..., -1, :-2]) * inv_dx2
            )
        if ny > 2:
            out[..., 1:-1, 0] = (
                psi[..., 2:, 0] - 2 * psi[..., 1:-1, 0] + psi[..., :-2, 0]
            ) * inv_dy2 + 2 * (psi[..., 1:-1, 1] - psi[..., 1:-1, 0]) * inv_dx2
            out[..., 1:-1, -1] = (
                psi[..., 2:, -1] - 2 * psi[..., 1:-1, -1] + psi[..., :-2, -1]
            ) * inv_dy2 + 2 * (psi[..., 1:-1, -2] - psi[..., 1:-1, -1]) * inv_dx2
        out[..., 0, 0] = (
            2 * (psi[..., 1, 0] - psi[..., 0, 0]) * inv_dy2
            + 2 * (psi[..., 0, 1] - psi[..., 0, 0]) * inv_dx2
        )
        out[..., 0, -1] = (
            2 * (psi[..., 1, -1] - psi[..., 0, -1]) * inv_dy2
            + 2 * (psi[..., 0, -2] - psi[..., 0, -1]) * inv_dx2
        )
        out[..., -1, 0] = (
            2 * (psi[..., -2, 0] - psi[..., -1, 0]) * inv_dy2
            + 2 * (psi[..., -1, 1] - psi[..., -1, 0]) * inv_dx2
        )
        out[..., -1, -1] = (
            2 * (psi[..., -2, -1] - psi[..., -1, -1]) * inv_dy2
            + 2 * (psi[..., -1, -2] - psi[..., -1, -1]) * inv_dx2
        )

    def _cpu_psi_rhs(self, psi, n_active, V, n_inactive, rho):
        """Compute the psi right-hand side on the CPU."""
        xp = self.xp
        lap = xp.empty_like(psi)
        self._cpu_laplacian(psi, lap)
        eff = V + self._g_C * rho + self._g_R * n_active + self._g_I * n_inactive
        gl = (self._R * n_active - self._gamma_C) * 0.5
        kinetic = self._neg_hbar2_over_2m * lap
        hbar_over_2m = -self._neg_hbar2_over_2m * self._inv_hbar
        relax = self._kinetic_relaxation_eta * n_active * hbar_over_2m
        return (-1j * self._inv_hbar) * (kinetic + eff * psi) + gl * psi + relax * lap

    def _step_cpu(self, V, pump, reservoir, boundary_condition, state):
        """Advance psi and the reservoir with one classic RK4 CPU step."""
        dt = self._dt
        t = state.t
        p0 = pump(t)
        p_half = pump(t + 0.5 * dt)
        p1 = pump(t + dt)
        psi0 = state.psi.copy()
        res0 = tuple(field.copy() for field in reservoir.get_state())

        def rhs(psi, res_state, pump_t):
            rho = self.xp.abs(psi) ** 2
            dpsi = self._cpu_psi_rhs(
                psi, reservoir.get_active_density(res_state), V,
                reservoir.get_inactive_density(res_state), rho,
            )
            dres = reservoir.get_derivatives(psi, pump_t, res_state, rho)
            return dpsi, dres

        def staged_reservoir(derivatives, scale):
            return tuple(r0 + scale * derivative for r0, derivative in zip(res0, derivatives))

        k1p, k1r = rhs(psi0, res0, p0)
        k2p, k2r = rhs(
            psi0 + 0.5 * dt * k1p, staged_reservoir(k1r, 0.5 * dt), p_half
        )
        k3p, k3r = rhs(
            psi0 + 0.5 * dt * k2p, staged_reservoir(k2r, 0.5 * dt), p_half
        )
        k4p, k4r = rhs(psi0 + dt * k3p, staged_reservoir(k3r, dt), p1)

        dt6 = dt / 6.0
        state.psi = boundary_condition.after_step_action(
            psi0 + dt6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        )
        reservoir.set_state(tuple(
            r0 + dt6 * (r1 + 2 * r2 + 2 * r3 + r4)
            for r0, r1, r2, r3, r4 in zip(res0, k1r, k2r, k3r, k4r)
        ))

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Callable[[float], Union[np.ndarray, cp.ndarray]],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        self.batch_physics.check(state.psi)
        if not self._use_gpu or state.psi.ndim == 3:
            self._step_cpu(potential, pump, reservoir, boundary_condition, state)
            return

        shape = state.psi.shape
        V_re_flat, V_im_flat = self._prepare_potential_buffers(potential)
        t = state.t
        dt = self._dt
        stage_times = (t, t + 0.5 * dt, t + dt)
        separable = pump.separable_term() if hasattr(pump, "separable_term") else None
        if separable is None:
            pumps = tuple(
                self._flat_gpu_input(pump(stage_time), self._real_dtype)
                for stage_time in stage_times
            )
            pump_scales = (self._np_real(1),) * 3
        else:
            envelope, coefficient = separable
            pump_flat = self._flat_gpu_input(envelope, self._real_dtype)
            pumps = (pump_flat,) * 3
            pump_scales = tuple(self._np_real(coefficient(stage_time))
                                for stage_time in stage_times)
        v_im_scale = self._np_real(1 if self.xp.iscomplexobj(potential) else 0)

        psi0 = self._flat_gpu_input(state.psi, self._complex_dtype)
        reservoir_state = reservoir.get_state()
        reservoir0 = tuple(
            self._flat_gpu_input(field, self._real_dtype)
            for field in reservoir_state
        )
        psi_out = self.xp.empty(self.N, dtype=self._complex_dtype)
        reservoir_out = tuple(
            self.xp.empty(self.N, dtype=self._real_dtype)
            for _ in reservoir0
        )

        # All producers, RK4 kernels, and consumers use the caller's current stream.
        self._step_gpu(
            psi0, reservoir0, V_re_flat, V_im_flat, pumps, pump_scales,
            v_im_scale, psi_out, reservoir_out
        )
        state.psi = boundary_condition.after_step_action(psi_out.reshape(shape))
        reservoir.set_state(tuple(field.reshape(shape) for field in reservoir_out))
