"""CUDA RK4 solver kernels and driver code."""
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
    from polarism.reservoir.abstract_reservoir import AbstractReservoir
    from polarism.grid.simulation_grid_2d import SimulationGrid2D

PREAMBLE_1D = """
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (tid >= N) return;
    const int i = tid / nx;
    const int j = tid % nx;
"""

COMBINE_PREAMBLE_1D = """
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
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
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    const {REAL} lap_nR = (nR[dn] - 2*nR_val + nR[up]) * inv_dy2
                        + (nR[rt] - 2*nR_val + nR[lt]) * inv_dx2;
    dnR_dt[tid] = pump[tid] - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;
}}
"""

_KERNEL_FUSED_STAGE_RHS_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void fused_stage_rhs(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k_nR,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    const int nx, const int ny,
    const {REAL} c_dt,
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
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    const {REAL} nR_up = nR0[up] + c_dt * k_nR[up];
    const {REAL} nR_dn = nR0[dn] + c_dt * k_nR[dn];
    const {REAL} nR_lt = nR0[lt] + c_dt * k_nR[lt];
    const {REAL} nR_rt = nR0[rt] + c_dt * k_nR[rt];
    const {REAL} lap_nR = (nR_dn - 2*nR_val + nR_up) * inv_dy2
                        + (nR_rt - 2*nR_val + nR_lt) * inv_dx2;
    dnR_dt[tid] = pump[tid] - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;
}}
"""

_KERNEL_COMBINE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void rk4_combine(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k1p, const {REAL}* __restrict__ k2p,
    const {REAL}* __restrict__ k3p, const {REAL}* __restrict__ k4p,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k1n, const {REAL}* __restrict__ k2n,
    const {REAL}* __restrict__ k3n, const {REAL}* __restrict__ k4n,
    {REAL}* __restrict__ psi_out,
    {REAL}* __restrict__ nR_out,
    const {REAL} dt6,
    const int nx, const int ny,
    const int N
) {{
    {COMBINE_PREAMBLE}
    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k3p[2*tid]   + k4p[2*tid]);
    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k3p[2*tid+1] + k4p[2*tid+1]);
    {REAL} nR_new = nR0[tid] + dt6 * (k1n[tid] + 2*k2n[tid] + 2*k3n[tid] + k4n[tid]);
    nR_out[tid] = (nR_new > 0) ? nR_new : 0;
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
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

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
    dnI_dt[tid] = pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val + diff_I * lap_nI;
    dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val + diff_A * lap_nA;
}}
"""

_KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void fused_stage_rhs_double(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nA0,
    const {REAL}* __restrict__ k_nA,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k_nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnA_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} c_dt,
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
    const {REAL} gl = (R_cond * nA_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nA_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

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

    dnI_dt[tid] = pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val + diff_I * lap_nI;
    dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val + diff_A * lap_nA;
}}
"""

_KERNEL_COMBINE_DOUBLE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void rk4_combine_double(
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
    const int nx, const int ny,
    const int N
) {{
    {COMBINE_PREAMBLE}
    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k3p[2*tid]   + k4p[2*tid]);
    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k3p[2*tid+1] + k4p[2*tid+1]);
    {REAL} nA_new = nA0[tid] + dt6 * (k1a[tid] + 2*k2a[tid] + 2*k3a[tid] + k4a[tid]);
    nA_out[tid] = (nA_new > 0) ? nA_new : 0;
    {REAL} nI_new = nI0[tid] + dt6 * (k1i[tid] + 2*k2i[tid] + 2*k3i[tid] + k4i[tid]);
    nI_out[tid] = (nI_new > 0) ? nI_new : 0;
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
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

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
    dnI_dt[tid] = pump[tid] - transfer - gamma_I * nI_val + diff_I * lap_nI;
}}
"""

_KERNEL_FUSED_STAGE_RHS_QDOUBLE_TEMPLATE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void fused_stage_rhs_qdouble(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k_psi,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k_nR,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k_nI,
    const {REAL}* __restrict__ V_re,
    const {REAL}* __restrict__ V_im,
    const {REAL}* __restrict__ pump,
    {REAL}* __restrict__ dpsi,
    {REAL}* __restrict__ dnR_dt,
    {REAL}* __restrict__ dnI_dt,
    const int nx, const int ny,
    const {REAL} c_dt,
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
    const {REAL} gl = (R_cond * nR_val - gamma_C) * ({REAL})0.5 + V_im[tid] * inv_hbar;

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = kinetic_relaxation_eta * nR_val * hbar_over_2m;

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

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

    dnR_dt[tid] = transfer - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;
    dnI_dt[tid] = pump[tid] - transfer - gamma_I * nI_val + diff_I * lap_nI;
}}
"""

_KERNEL_COMBINE_QDOUBLE = r"""
extern "C" __global__ {LAUNCH_BOUNDS} void rk4_combine_qdouble(
    const {REAL}* __restrict__ psi0,
    const {REAL}* __restrict__ k1p, const {REAL}* __restrict__ k2p,
    const {REAL}* __restrict__ k3p, const {REAL}* __restrict__ k4p,
    const {REAL}* __restrict__ nR0,
    const {REAL}* __restrict__ k1r, const {REAL}* __restrict__ k2r,
    const {REAL}* __restrict__ k3r, const {REAL}* __restrict__ k4r,
    const {REAL}* __restrict__ nI0,
    const {REAL}* __restrict__ k1i, const {REAL}* __restrict__ k2i,
    const {REAL}* __restrict__ k3i, const {REAL}* __restrict__ k4i,
    {REAL}* __restrict__ psi_out,
    {REAL}* __restrict__ nR_out,
    {REAL}* __restrict__ nI_out,
    const {REAL} dt6,
    const int nx, const int ny,
    const int N
) {{
    {COMBINE_PREAMBLE}
    psi_out[2*tid]   = psi0[2*tid]   + dt6 * (k1p[2*tid]   + 2*k2p[2*tid]   + 2*k3p[2*tid]   + k4p[2*tid]);
    psi_out[2*tid+1] = psi0[2*tid+1] + dt6 * (k1p[2*tid+1] + 2*k2p[2*tid+1] + 2*k3p[2*tid+1] + k4p[2*tid+1]);
    {REAL} nR_new = nR0[tid] + dt6 * (k1r[tid] + 2*k2r[tid] + 2*k3r[tid] + k4r[tid]);
    nR_out[tid] = (nR_new > 0) ? nR_new : 0;
    {REAL} nI_new = nI0[tid] + dt6 * (k1i[tid] + 2*k2i[tid] + 2*k3i[tid] + k4i[tid]);
    nI_out[tid] = (nI_new > 0) ? nI_new : 0;
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
    combine_preamble: str = COMBINE_PREAMBLE_1D,
    launch_bounds: str = "",
) -> tuple[str, str, str]:
    """Build the 1D CUDA kernel source."""
    neighbor_code = (
        _NEUMANN_NEIGHBORS if bc_type == "closed-interval" else _PERIODIC_NEIGHBORS
    )
    if reservoir_type == "double":
        rhs_tmpl = _KERNEL_RHS_DOUBLE_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_DOUBLE_TEMPLATE
        combine_tmpl = _KERNEL_COMBINE_DOUBLE
    elif reservoir_type == "quadratic-double":
        rhs_tmpl = _KERNEL_RHS_QDOUBLE_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_QDOUBLE_TEMPLATE
        combine_tmpl = _KERNEL_COMBINE_QDOUBLE
    else:
        rhs_tmpl = _KERNEL_RHS_TEMPLATE
        fused_tmpl = _KERNEL_FUSED_STAGE_RHS_TEMPLATE
        combine_tmpl = _KERNEL_COMBINE

    diag_neighbors, extra_psi_diag, lap_block_rhs, lap_block_fused = _build_laplacian_code(
        laplacian_type, bc_type, real_type
    )
    fmt = dict(
        REAL=real_type,
        NEIGHBOR_CODE=neighbor_code,
        PREAMBLE=preamble,
        COMBINE_PREAMBLE=combine_preamble,
        LAUNCH_BOUNDS=launch_bounds,
        DIAG_NEIGHBORS=diag_neighbors,
        EXTRA_PSI_DIAG=extra_psi_diag,
        LAP_BLOCK_RHS=lap_block_rhs,
        LAP_BLOCK_FUSED=lap_block_fused,
    )
    return (
        rhs_tmpl.format(**fmt),
        fused_tmpl.format(**fmt),
        combine_tmpl.format(**fmt),
    )


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

    def __init__(self, config: Config, grid: SimulationGrid2D):
        """Set up the rk 4 cuda solver."""
        super().__init__(config)
        self.nx = grid.nx
        self.ny = grid.ny
        self.N = grid.nx * grid.ny
        self.grid_type = getattr(config.grid, "grid_type", "periodic")
        _res_cfg = getattr(config, "reservoir", None)
        self._reservoir_type = (
            getattr(_res_cfg, "reservoir_type", "single") if _res_cfg else "single"
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

        self._hbar = config.physics.hbar
        self._m_eff = config.physics.m_eff
        self._g_C = config.physics.g_C
        self._g_R = config.physics.g_R
        self._g_I = getattr(config.physics, "g_I", 0.0)
        self._R = config.physics.R
        self._gamma_C = config.physics.gamma_C
        self._gamma_R = config.physics.gamma_R
        self._dt = config.solver.dt
        self._kinetic_relaxation_eta = getattr(config.physics, "kinetic_relaxation_eta", 0.0)
        self._reservoir_diffusion_I = getattr(config.physics, "reservoir_diffusion_I", 0.0)
        self._reservoir_diffusion_A = getattr(config.physics, "reservoir_diffusion_A", 0.0)
        self._reservoir_diffusion_R = getattr(config.physics, "reservoir_diffusion_R", 0.0)

        if self._reservoir_type == "double":
            self._gamma_I = config.physics.gamma_I
            self._gamma_A = config.physics.gamma_A
            self._R_IA = config.physics.R_IA
            self._R_AI = config.physics.R_AI
        elif self._reservoir_type == "quadratic-double":
            self._gamma_I = config.physics.gamma_I
            self._kappa = config.physics.kappa

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

        self._use_gpu = hasattr(self.xp, "cuda")

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

            import cupy as _cp
            self._stream_compute = _cp.cuda.Stream(non_blocking=True)
            self._stream_transfer = _cp.cuda.Stream(non_blocking=True)
            self._transfer_event = _cp.cuda.Event()

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

        rhs_src, fused_src, combine_src = self._build_kernel_source(
            self.grid_type, self._real_type, self._reservoir_type
        )

        if self._reservoir_type == "double":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_double")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_double")
            self._kern_combine = cp.RawKernel(combine_src, "rk4_combine_double")
        elif self._reservoir_type == "quadratic-double":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_qdouble")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_qdouble")
            self._kern_combine = cp.RawKernel(combine_src, "rk4_combine_qdouble")
        else:
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs")
            self._kern_combine = cp.RawKernel(combine_src, "rk4_combine")

        block = self._select_block_size()
        self._make_launch_dims(block)

    def _init_gpu_buffers(self) -> None:
        """Allocate GPU work buffers."""
        xp = self.xp
        N = self.N

        cdtype = xp.complex64 if self._precision == "single" else xp.complex128
        rdtype = xp.float32 if self._precision == "single" else xp.float64

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

        self._V_im_zero = xp.zeros(N, dtype=rdtype)

        self._potential_cache_key: tuple | None = None
        self._V_re_flat = None
        self._V_im_flat = None

        if self._reservoir_type in ("double", "quadratic-double"):
            self._k1_nI = xp.empty(N, dtype=rdtype)
            self._k2_nI = xp.empty(N, dtype=rdtype)
            self._k3_nI = xp.empty(N, dtype=rdtype)
            self._k4_nI = xp.empty(N, dtype=rdtype)
            self._nI_tmp = xp.empty(N, dtype=rdtype)
            self._nI0 = xp.empty(N, dtype=rdtype)

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
        key = (id(potential), potential.shape, potential.dtype)
        if key == self._potential_cache_key:
            return self._V_re_flat, self._V_im_flat

        if xp.iscomplexobj(potential):
            V_re = self._cast_to_precision(xp.ascontiguousarray(xp.real(potential)))
            V_im = self._cast_to_precision(xp.ascontiguousarray(xp.imag(potential)))
            self._V_re_flat = V_re.ravel()
            self._V_im_flat = V_im.ravel()
        else:
            V_re = potential if potential.flags["C_CONTIGUOUS"] else xp.ascontiguousarray(potential)
            self._V_re_flat = self._cast_to_precision(V_re).ravel()
            self._V_im_flat = self._V_im_zero

        self._potential_cache_key = key
        return self._V_re_flat, self._V_im_flat

    def _gpu_rhs(self, psi, nR, V_re, V_im, pump, out_dpsi, out_dnR):
        """Compute the right-hand side."""
        self._kern_rhs(
            self._k_grid, self._k_block,
            (
                psi, nR, V_re, V_im, pump, out_dpsi, out_dnR,
                self._k_nx, self._k_ny,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C, self._k_gamma_R,
                self._k_kinetic_relaxation_eta, self._k_diff_R,
            ),
        )

    def _gpu_fused_stage_rhs(self, psi0, k_psi, nR0, k_nR, c_dt, V_re, V_im, pump, out_dpsi, out_dnR):
        """Compute one staged right-hand side on the GPU."""
        self._kern_fused(
            self._k_grid, self._k_block,
            (
                psi0, k_psi, nR0, k_nR, V_re, V_im, pump, out_dpsi, out_dnR,
                self._k_nx, self._k_ny,
                c_dt,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
                self._k_R, self._k_gamma_C, self._k_gamma_R,
                self._k_kinetic_relaxation_eta, self._k_diff_R,
            ),
        )

    def _gpu_combine(
        self, psi0, k1p, k2p, k3p, k4p, nR0, k1n, k2n, k3n, k4n, psi_out, nR_out
    ):
        """Combine RK4 stages on the GPU."""
        self._kern_combine(
            self._k_grid, self._k_block,
            (
                psi0, k1p, k2p, k3p, k4p,
                nR0, k1n, k2n, k3n, k4n,
                psi_out, nR_out,
                self._k_dt6,
                self._k_nx, self._k_ny, self._k_N,
            ),
        )

    def _step_gpu(self, V_re, V_im, pump, nR_buf, psi_buf):
        """Run one RK4 step on the GPU."""
        psi0 = self._psi0
        nR0 = self._nA0

        psi0[:] = psi_buf
        nR0[:] = nR_buf

        self._gpu_rhs(psi0, nR0, V_re, V_im, pump, self._k1_psi, self._k1_nA)

        self._gpu_fused_stage_rhs(
            psi0, self._k1_psi, nR0, self._k1_nA,
            self._k_half_dt, V_re, V_im, pump, self._k2_psi, self._k2_nA,
        )
        self._gpu_fused_stage_rhs(
            psi0, self._k2_psi, nR0, self._k2_nA,
            self._k_half_dt, V_re, V_im, pump, self._k3_psi, self._k3_nA,
        )
        self._gpu_fused_stage_rhs(
            psi0, self._k3_psi, nR0, self._k3_nA,
            self._k_dt, V_re, V_im, pump, self._k4_psi, self._k4_nA,
        )

        self._gpu_combine(
            psi0, self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi,
            nR0, self._k1_nA, self._k2_nA, self._k3_nA, self._k4_nA,
            psi_buf, nR_buf,
        )

    def _gpu_rhs_double(self, psi, nA, nI, V_re, V_im, pump, out_dpsi, out_dnA, out_dnI):
        """Compute the right-hand side."""
        self._kern_rhs(
            self._k_grid, self._k_block,
            (
                psi, nA, nI, V_re, V_im, pump, out_dpsi, out_dnA, out_dnI,
                self._k_nx, self._k_ny,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R, self._k_g_I,
                self._k_R, self._k_gamma_C,
                self._k_gamma_I, self._k_gamma_A,
                self._k_R_IA, self._k_R_AI,
                self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_A,
            ),
        )

    def _gpu_fused_stage_rhs_double(
        self, psi0, k_psi, nA0, k_nA, nI0, k_nI, c_dt, V_re, V_im, pump,
        out_dpsi, out_dnA, out_dnI,
    ):
        """Compute one staged right-hand side on the GPU."""
        self._kern_fused(
            self._k_grid, self._k_block,
            (
                psi0, k_psi, nA0, k_nA, nI0, k_nI, V_re, V_im, pump,
                out_dpsi, out_dnA, out_dnI,
                self._k_nx, self._k_ny,
                c_dt,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R, self._k_g_I,
                self._k_R, self._k_gamma_C,
                self._k_gamma_I, self._k_gamma_A,
                self._k_R_IA, self._k_R_AI,
                self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_A,
            ),
        )

    def _gpu_combine_double(
        self, psi0, k1p, k2p, k3p, k4p,
        nA0, k1a, k2a, k3a, k4a,
        nI0, k1i, k2i, k3i, k4i,
        psi_out, nA_out, nI_out,
    ):
        """Combine RK4 stages on the GPU."""
        self._kern_combine(
            self._k_grid, self._k_block,
            (
                psi0, k1p, k2p, k3p, k4p,
                nA0, k1a, k2a, k3a, k4a,
                nI0, k1i, k2i, k3i, k4i,
                psi_out, nA_out, nI_out,
                self._k_dt6,
                self._k_nx, self._k_ny, self._k_N,
            ),
        )

    def _step_gpu_double(self, V_re, V_im, pump, nA_buf, nI_buf, psi_buf):
        """Run one RK4 step on the GPU."""
        psi0 = self._psi0
        nA0 = self._nA0
        nI0 = self._nI0

        psi0[:] = psi_buf
        nA0[:] = nA_buf
        nI0[:] = nI_buf

        self._gpu_rhs_double(psi0, nA0, nI0, V_re, V_im, pump,
                              self._k1_psi, self._k1_nA, self._k1_nI)

        self._gpu_fused_stage_rhs_double(
            psi0, self._k1_psi, nA0, self._k1_nA, nI0, self._k1_nI,
            self._k_half_dt, V_re, V_im, pump,
            self._k2_psi, self._k2_nA, self._k2_nI,
        )
        self._gpu_fused_stage_rhs_double(
            psi0, self._k2_psi, nA0, self._k2_nA, nI0, self._k2_nI,
            self._k_half_dt, V_re, V_im, pump,
            self._k3_psi, self._k3_nA, self._k3_nI,
        )
        self._gpu_fused_stage_rhs_double(
            psi0, self._k3_psi, nA0, self._k3_nA, nI0, self._k3_nI,
            self._k_dt, V_re, V_im, pump,
            self._k4_psi, self._k4_nA, self._k4_nI,
        )

        self._gpu_combine_double(
            psi0,
            self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi,
            nA0,
            self._k1_nA, self._k2_nA, self._k3_nA, self._k4_nA,
            nI0,
            self._k1_nI, self._k2_nI, self._k3_nI, self._k4_nI,
            psi_buf, nA_buf, nI_buf,
        )

    def _gpu_rhs_qdouble(self, psi, nR, nI, V_re, V_im, pump, out_dpsi, out_dnR, out_dnI):
        """Compute the right-hand side for quadratic-double reservoir."""
        self._kern_rhs(
            self._k_grid, self._k_block,
            (
                psi, nR, nI, V_re, V_im, pump, out_dpsi, out_dnR, out_dnI,
                self._k_nx, self._k_ny,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R, self._k_g_I,
                self._k_R, self._k_gamma_C,
                self._k_gamma_R, self._k_gamma_I, self._k_kappa,
                self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_R,
            ),
        )

    def _gpu_fused_stage_rhs_qdouble(
        self, psi0, k_psi, nR0, k_nR, nI0, k_nI, c_dt, V_re, V_im, pump,
        out_dpsi, out_dnR, out_dnI,
    ):
        """Compute one staged right-hand side for quadratic-double on the GPU."""
        self._kern_fused(
            self._k_grid, self._k_block,
            (
                psi0, k_psi, nR0, k_nR, nI0, k_nI, V_re, V_im, pump,
                out_dpsi, out_dnR, out_dnI,
                self._k_nx, self._k_ny,
                c_dt,
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R, self._k_g_I,
                self._k_R, self._k_gamma_C,
                self._k_gamma_R, self._k_gamma_I, self._k_kappa,
                self._k_kinetic_relaxation_eta, self._k_diff_I, self._k_diff_R,
            ),
        )

    def _gpu_combine_qdouble(
        self, psi0, k1p, k2p, k3p, k4p,
        nR0, k1r, k2r, k3r, k4r,
        nI0, k1i, k2i, k3i, k4i,
        psi_out, nR_out, nI_out,
    ):
        """Combine RK4 stages for quadratic-double on the GPU."""
        self._kern_combine(
            self._k_grid, self._k_block,
            (
                psi0, k1p, k2p, k3p, k4p,
                nR0, k1r, k2r, k3r, k4r,
                nI0, k1i, k2i, k3i, k4i,
                psi_out, nR_out, nI_out,
                self._k_dt6,
                self._k_nx, self._k_ny, self._k_N,
            ),
        )

    def _step_gpu_qdouble(self, V_re, V_im, pump, nR_buf, nI_buf, psi_buf):
        """Run one RK4 step on the GPU for quadratic-double reservoir."""
        psi0 = self._psi0
        nR0 = self._nA0
        nI0 = self._nI0

        psi0[:] = psi_buf
        nR0[:] = nR_buf
        nI0[:] = nI_buf

        self._gpu_rhs_qdouble(psi0, nR0, nI0, V_re, V_im, pump,
                               self._k1_psi, self._k1_nA, self._k1_nI)

        self._gpu_fused_stage_rhs_qdouble(
            psi0, self._k1_psi, nR0, self._k1_nA, nI0, self._k1_nI,
            self._k_half_dt, V_re, V_im, pump,
            self._k2_psi, self._k2_nA, self._k2_nI,
        )
        self._gpu_fused_stage_rhs_qdouble(
            psi0, self._k2_psi, nR0, self._k2_nA, nI0, self._k2_nI,
            self._k_half_dt, V_re, V_im, pump,
            self._k3_psi, self._k3_nA, self._k3_nI,
        )
        self._gpu_fused_stage_rhs_qdouble(
            psi0, self._k3_psi, nR0, self._k3_nA, nI0, self._k3_nI,
            self._k_dt, V_re, V_im, pump,
            self._k4_psi, self._k4_nA, self._k4_nI,
        )

        self._gpu_combine_qdouble(
            psi0,
            self._k1_psi, self._k2_psi, self._k3_psi, self._k4_psi,
            nR0,
            self._k1_nA, self._k2_nA, self._k3_nA, self._k4_nA,
            nI0,
            self._k1_nI, self._k2_nI, self._k3_nI, self._k4_nI,
            psi_buf, nR_buf, nI_buf,
        )

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
            self._cpu_laplacian_9pt(psi, out)
            return
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

    def _cpu_psi_rhs(self, psi, n_active, V, n_inactive=None):
        """Compute the psi right-hand side on the CPU."""
        xp = self.xp
        lap = xp.empty_like(psi)
        self._cpu_laplacian(psi, lap)
        rho = xp.abs(psi) ** 2
        if n_inactive is None:
            n_inactive = n_active * 0.0
        eff = V + self._g_C * rho + self._g_R * n_active + self._g_I * n_inactive
        gl = (self._R * n_active - self._gamma_C) * 0.5
        kinetic = self._neg_hbar2_over_2m * lap
        hbar_over_2m = -self._neg_hbar2_over_2m * self._inv_hbar
        relax = self._kinetic_relaxation_eta * n_active * hbar_over_2m
        return (-1j * self._inv_hbar) * (kinetic + eff * psi) + gl * psi + relax * lap

    def _cpu_laplacian_real(self, n):
        """Compute Laplacian of a real-valued 2D field on the CPU."""
        xp = self.xp
        out = xp.empty_like(n)
        inv_dx2 = self._inv_dx2
        inv_dy2 = self._inv_dy2
        if self.grid_type == "periodic":
            out[:] = (
                xp.roll(n, -1, axis=0) - 2 * n + xp.roll(n, 1, axis=0)
            ) * inv_dy2 + (
                xp.roll(n, -1, axis=1) - 2 * n + xp.roll(n, 1, axis=1)
            ) * inv_dx2
            return out
        ny, nx = self.ny, self.nx
        out.fill(0)
        if ny > 2 and nx > 2:
            out[1:-1, 1:-1] = (
                n[2:, 1:-1] - 2 * n[1:-1, 1:-1] + n[:-2, 1:-1]
            ) * inv_dy2 + (
                n[1:-1, 2:] - 2 * n[1:-1, 1:-1] + n[1:-1, :-2]
            ) * inv_dx2
        if nx > 2:
            out[0, 1:-1] = (
                2 * (n[1, 1:-1] - n[0, 1:-1]) * inv_dy2
                + (n[0, 2:] - 2 * n[0, 1:-1] + n[0, :-2]) * inv_dx2
            )
            out[-1, 1:-1] = (
                2 * (n[-2, 1:-1] - n[-1, 1:-1]) * inv_dy2
                + (n[-1, 2:] - 2 * n[-1, 1:-1] + n[-1, :-2]) * inv_dx2
            )
        if ny > 2:
            out[1:-1, 0] = (
                n[2:, 0] - 2 * n[1:-1, 0] + n[:-2, 0]
            ) * inv_dy2 + 2 * (n[1:-1, 1] - n[1:-1, 0]) * inv_dx2
            out[1:-1, -1] = (
                n[2:, -1] - 2 * n[1:-1, -1] + n[:-2, -1]
            ) * inv_dy2 + 2 * (n[1:-1, -2] - n[1:-1, -1]) * inv_dx2
        out[0, 0] = (
            2 * (n[1, 0] - n[0, 0]) * inv_dy2 + 2 * (n[0, 1] - n[0, 0]) * inv_dx2
        )
        out[0, -1] = (
            2 * (n[1, -1] - n[0, -1]) * inv_dy2 + 2 * (n[0, -2] - n[0, -1]) * inv_dx2
        )
        out[-1, 0] = (
            2 * (n[-2, 0] - n[-1, 0]) * inv_dy2 + 2 * (n[-1, 1] - n[-1, 0]) * inv_dx2
        )
        out[-1, -1] = (
            2 * (n[-2, -1] - n[-1, -1]) * inv_dy2 + 2 * (n[-1, -2] - n[-1, -1]) * inv_dx2
        )
        return out

    def _cpu_rhs(self, psi, nR, V, pump):
        """Compute the right-hand side."""
        dpsi = self._cpu_psi_rhs(psi, nR, V)
        rho = self.xp.abs(psi) ** 2
        dnR = (
            pump - (self._gamma_R + self._R * rho) * nR
            + self._reservoir_diffusion_R * self._cpu_laplacian_real(nR)
        )
        return dpsi, dnR

    def _cpu_rhs_double(self, psi, nA, nI, V, pump):
        """Compute the right-hand side."""
        dpsi = self._cpu_psi_rhs(psi, nA, V, nI)
        rho = self.xp.abs(psi) ** 2
        dnI = (
            pump - (self._gamma_I + self._R_IA) * nI + self._R_AI * nA
            + self._reservoir_diffusion_I * self._cpu_laplacian_real(nI)
        )
        dnA = (
            self._R_IA * nI - (self._gamma_A + self._R_AI + self._R * rho) * nA
            + self._reservoir_diffusion_A * self._cpu_laplacian_real(nA)
        )
        return dpsi, dnA, dnI

    def _cpu_rhs_quadratic_double(self, psi, nR, nI, V, pump):
        """Compute the right-hand side for quadratic-double reservoir."""
        dpsi = self._cpu_psi_rhs(psi, nR, V, nI)
        rho = self.xp.abs(psi) ** 2
        transfer = self._kappa * nI ** 2
        dnR = (
            transfer - (self._gamma_R + self._R * rho) * nR
            + self._reservoir_diffusion_R * self._cpu_laplacian_real(nR)
        )
        dnI = (
            pump - transfer - self._gamma_I * nI
            + self._reservoir_diffusion_I * self._cpu_laplacian_real(nI)
        )
        return dpsi, dnR, dnI

    def _step_cpu_quadratic_double(self, V, pump, reservoir, boundary_condition, state):
        """Run one RK4 step on the CPU for quadratic-double reservoir."""
        dt = self._dt
        xp = self.xp
        psi0 = state.psi.copy()
        nR0, nI0 = reservoir.get_state()
        nR0 = nR0.copy()
        nI0 = nI0.copy()

        k1p, k1r, k1i = self._cpu_rhs_quadratic_double(psi0, nR0, nI0, V, pump)
        k2p, k2r, k2i = self._cpu_rhs_quadratic_double(
            psi0 + 0.5 * dt * k1p, nR0 + 0.5 * dt * k1r, nI0 + 0.5 * dt * k1i, V, pump
        )
        k3p, k3r, k3i = self._cpu_rhs_quadratic_double(
            psi0 + 0.5 * dt * k2p, nR0 + 0.5 * dt * k2r, nI0 + 0.5 * dt * k2i, V, pump
        )
        k4p, k4r, k4i = self._cpu_rhs_quadratic_double(
            psi0 + dt * k3p, nR0 + dt * k3r, nI0 + dt * k3i, V, pump
        )

        dt6 = dt / 6.0
        state.psi = psi0 + dt6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        state.psi = boundary_condition.after_step_action(state.psi)
        nR_new = nR0 + dt6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        nI_new = nI0 + dt6 * (k1i + 2 * k2i + 2 * k3i + k4i)
        reservoir.set_state((xp.maximum(nR_new, 0), xp.maximum(nI_new, 0)))

    def _step_cpu(self, V, pump, reservoir, boundary_condition, state):
        """Run one RK4 step on the CPU."""
        dt = self._dt
        psi0 = state.psi.copy()
        nR0 = reservoir.get_reservoir_density().copy()

        k1p, k1n = self._cpu_rhs(psi0, nR0, V, pump)
        k2p, k2n = self._cpu_rhs(psi0 + 0.5 * dt * k1p, nR0 + 0.5 * dt * k1n, V, pump)
        k3p, k3n = self._cpu_rhs(psi0 + 0.5 * dt * k2p, nR0 + 0.5 * dt * k2n, V, pump)
        k4p, k4n = self._cpu_rhs(psi0 + dt * k3p, nR0 + dt * k3n, V, pump)

        state.psi = psi0 + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        state.psi = boundary_condition.after_step_action(state.psi)
        nR_new = nR0 + (dt / 6.0) * (k1n + 2 * k2n + 2 * k3n + k4n)
        reservoir.set_state((nR_new,))

    def _step_cpu_double(self, V, pump, reservoir, boundary_condition, state):
        """Run one RK4 step on the CPU."""
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

    def step(
        self,
        potential: Union[np.ndarray, cp.ndarray],
        pump: Union[np.ndarray, cp.ndarray],
        reservoir: AbstractReservoir,
        boundary_condition: BoundaryCondition,
        state: SimulationState,
    ) -> None:
        """Advance the solver by one time step."""
        if not self._use_gpu:
            if self._reservoir_type == "double":
                self._step_cpu_double(potential, pump, reservoir, boundary_condition, state)
            elif self._reservoir_type == "quadratic-double":
                self._step_cpu_quadratic_double(potential, pump, reservoir, boundary_condition, state)
            else:
                self._step_cpu(potential, pump, reservoir, boundary_condition, state)
            return

        shape = state.psi.shape

        V_re_flat, V_im_flat = self._prepare_potential_buffers(potential)
        pump_flat = self._cast_to_precision(pump).ravel()

        with self._stream_compute:
            if self._reservoir_type == "double":
                nA, nI = reservoir.get_state()
                psi_buf = self._psi_tmp
                psi_buf[:] = self._cast_to_precision(state.psi).ravel()
                nA_buf = self._nA_tmp
                nA_buf[:] = self._cast_to_precision(nA).ravel()
                nI_buf = self._nI_tmp
                nI_buf[:] = self._cast_to_precision(nI).ravel()

                self._step_gpu_double(V_re_flat, V_im_flat, pump_flat, nA_buf, nI_buf, psi_buf)
                self._transfer_event.record(self._stream_compute)
            elif self._reservoir_type == "quadratic-double":
                nR, nI = reservoir.get_state()
                psi_buf = self._psi_tmp
                psi_buf[:] = self._cast_to_precision(state.psi).ravel()
                nR_buf = self._nA_tmp
                nR_buf[:] = self._cast_to_precision(nR).ravel()
                nI_buf = self._nI_tmp
                nI_buf[:] = self._cast_to_precision(nI).ravel()

                self._step_gpu_qdouble(V_re_flat, V_im_flat, pump_flat, nR_buf, nI_buf, psi_buf)
                self._transfer_event.record(self._stream_compute)
            else:
                psi_buf = self._psi_tmp
                psi_buf[:] = self._cast_to_precision(state.psi).ravel()
                nR_buf = self._nA_tmp
                nR_buf[:] = self._cast_to_precision(
                    reservoir.get_reservoir_density()
                ).ravel()

                self._step_gpu(V_re_flat, V_im_flat, pump_flat, nR_buf, psi_buf)
                self._transfer_event.record(self._stream_compute)

        self._stream_transfer.wait_event(self._transfer_event)
        with self._stream_transfer:
            if self._reservoir_type == "double":
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
            elif self._reservoir_type == "quadratic-double":
                psi_result = psi_buf.reshape(shape)
                if state.psi.dtype != psi_result.dtype:
                    psi_result = psi_result.astype(state.psi.dtype)
                state.psi = psi_result
                state.psi = boundary_condition.after_step_action(state.psi)

                res_shape = (self.ny, self.nx)
                nR_result = nR_buf.reshape(res_shape)
                nI_result = nI_buf.reshape(res_shape)
                if nR.dtype != nR_result.dtype:
                    nR_result = nR_result.astype(nR.dtype)
                    nI_result = nI_result.astype(nI.dtype)
                reservoir.set_state((
                    self.xp.maximum(nR_result, 0),
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

        self._stream_transfer.synchronize()
