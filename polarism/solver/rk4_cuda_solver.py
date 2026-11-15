"""CUDA RK4 solver kernels and driver code."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Union

from polarism.compute_engine import has_full_rate_fp64
from polarism.laser.pump_field import evaluate_pump
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver, ACCEPTED, SolverCapabilities
from polarism.solver.solver_registry import register_solver

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.boundary_conditions.boundary_condition import BoundaryCondition
    from polarism.config.simulation_parameters import Config
    from polarism.reservoir.abstract_reservoir import AbstractReservoir
    from polarism.grid.simulation_grid import SimulationGrid

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
    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    {RESERVOIR_RHS}
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
    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

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

    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    {RESERVOIR_RHS}
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
    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

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

    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

    dpsi[2*tid]   =  inv_hbar * A_im + gl * pre + relax * lap_re;
    dpsi[2*tid+1] = -inv_hbar * A_re + gl * pim + relax * lap_im;

    {RESERVOIR_RHS}
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

    const {REAL} eff = {ENERGY_EXPRESSION};
    const {REAL} gl = {RATE_EXPRESSION};

    const {REAL} A_re = neg_hbar2_over_2m * lap_re + eff * pre;
    const {REAL} A_im = neg_hbar2_over_2m * lap_im + eff * pim;

    const {REAL} hbar_over_2m = -neg_hbar2_over_2m * inv_hbar;
    const {REAL} relax = {RELAX_EXPRESSION};

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

def _reservoir_cuda_snippet(reservoir_type: str, real_type: str):
    """Assemble the reservoir stage equations in their original operation order."""
    lines = {
        "single": (
            "const {REAL} lap_nR = (nR[dn] - 2*nR_val + nR[up]) * inv_dy2",
            "                    + (nR[rt] - 2*nR_val + nR[lt]) * inv_dx2;",
            "dnR_dt[tid] = pump_scale * pump[tid] - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;",
        ),
        "double": (
            "const {REAL} lap_nI = (nI[dn] - 2*nI_val + nI[up]) * inv_dy2",
            "                    + (nI[rt] - 2*nI_val + nI[lt]) * inv_dx2;",
            "const {REAL} lap_nA = (nA[dn] - 2*nA_val + nA[up]) * inv_dy2",
            "                    + (nA[rt] - 2*nA_val + nA[lt]) * inv_dx2;",
            "dnI_dt[tid] = pump_scale * pump[tid] - (gamma_I + R_IA) * nI_val + R_AI * nA_val + diff_I * lap_nI;",
            "dnA_dt[tid] = R_IA * nI_val - (gamma_A + R_AI + R_cond * rho) * nA_val + diff_A * lap_nA;",
        ),
        "quadratic-double": (
            "const {REAL} lap_nR = (nR[dn] - 2*nR_val + nR[up]) * inv_dy2",
            "                    + (nR[rt] - 2*nR_val + nR[lt]) * inv_dx2;",
            "const {REAL} lap_nI = (nI[dn] - 2*nI_val + nI[up]) * inv_dy2",
            "                    + (nI[rt] - 2*nI_val + nI[lt]) * inv_dx2;",
            "dnR_dt[tid] = transfer - (gamma_R + R_cond * rho) * nR_val + diff_R * lap_nR;",
            "dnI_dt[tid] = pump_scale * pump[tid] - transfer - gamma_I * nI_val + diff_I * lap_nI;",
        ),
    }[reservoir_type]
    return ("\n    ".join(lines)).replace("{REAL}", real_type)


def _core_cuda_snippets(reservoir_type: str, real_type: str):
    """Return ordered core energy, rate, and reservoir expressions."""
    energy = ("V_re[tid]", "g_C * rho")
    active = "nA_val" if reservoir_type == "double" else "nR_val"
    energy += (f"g_R * {active}",)
    if reservoir_type != "single":
        energy += ("g_I * nI_val",)
    rate = f"(R_cond * {active} - gamma_C) * ({real_type})0.5 + v_im_scale * V_im[tid] * inv_hbar"
    relax = f"kinetic_relaxation_eta * {active} * hbar_over_2m"
    return " + ".join(energy), rate, relax


def _wigner_arguments(real_type: str, graph: bool) -> str:
    """Return the inputs shared by flat and tiled Wigner kernels."""
    step_decl = ("const unsigned long long* __restrict__ noise_step_base, "
                 "const unsigned int noise_step_offset" if graph else
                 "const unsigned long long noise_step")
    return (
        "const unsigned long long* __restrict__ noise_seeds, "
        f"const {real_type}* __restrict__ noise_gamma, "
        f"const {real_type}* __restrict__ noise_scattering, "
        f"const {real_type}* __restrict__ noise_strength, "
        f"const {real_type} noise_dt_over_4v, {step_decl}"
    )


def _wigner_increment(real_type: str, active: str, member: str, cell: str,
                      output_index: str, graph: bool, tiled: bool = False) -> str:
    """Generate the same Philox and Box-Muller increment for either RK4 path."""
    precision = "single" if real_type == "float" else "double"
    step = ("noise_step_base[0] + noise_step_offset" if graph else "noise_step")
    if precision == "single":
        uniforms = (
            "float u1 = (float)((a >> 8) + 1U) * 0x1p-24f; "
            "float u2 = (float)((b >> 8) + 1U) * 0x1p-24f;"
        )
        sqrt_fn, log_fn, sincos_fn = "sqrtf", "logf", "sincosf"
    else:
        uniforms = (
            "double u1 = ((double)(a >> 5) * 67108864.0 + (double)(b >> 6) + 1.0) * 0x1p-53; "
            "double u2 = ((double)(c >> 5) * 67108864.0 + (double)(d >> 6) + 1.0) * 0x1p-53;"
        )
        sqrt_fn, log_fn, sincos_fn = "sqrt", "log", "sincos"
    output_re = (f"psi_out[2*{output_index}] = psi_final_re +"
                 if tiled else f"psi_out[2*{output_index}] +=")
    output_im = (f"psi_out[2*{output_index}+1] = psi_final_im +"
                 if tiled else f"psi_out[2*{output_index}+1] +=")
    return f"""
    const int noise_member = {member};
    const unsigned long long seed = noise_seeds[noise_member];
    const unsigned long long step = {step};
    unsigned int a = (unsigned int)({cell});
    unsigned int b = (unsigned int)step;
    unsigned int c = (unsigned int)(step >> 32);
    unsigned int d = 1U;
    philox4x32(&a, &b, &c, &d, (unsigned int)seed, (unsigned int)(seed >> 32));
    {uniforms}
    {real_type} radius = {sqrt_fn}(({real_type})(-2) * {log_fn}(u1));
    {real_type} angle = ({real_type})6.283185307179586476925286766559 * u2;
    {real_type} sn, cs;
    {sincos_fn}(angle, &sn, &cs);
    {real_type} sigma = {sqrt_fn}(noise_strength[noise_member] *
        (noise_gamma[noise_member] + noise_scattering[noise_member] * {active})
        * noise_dt_over_4v);
    {output_re} sigma * radius * cs;
    {output_im} sigma * radius * sn;
"""


def _fused_noise_source(source: str, real_type: str, reservoir_type: str,
                        batched: bool, graph: bool) -> str:
    """Append the Philox Wigner increment to a final-stage flat kernel."""
    from polarism.noise.philox import cuda_source

    precision = "single" if real_type == "float" else "double"
    philox = cuda_source(precision).split('extern "C" __global__ void philox_words', 1)[0]
    header, body = source.split(") {", 1)
    header += ", " + _wigner_arguments(real_type, graph)
    active = {"single": "nR_out[tid]", "double": "nA_out[tid]",
              "quadratic-double": "nR_out[tid]", "none": f"({real_type})0"}[reservoir_type]
    member = "member" if batched else "0"
    cell = "spatial" if batched else "tid"
    increment = _wigner_increment(real_type, active, member, cell, "tid", graph)
    return philox + header + ") {" + body.rsplit("\n}", 1)[0] + increment + "}\n"


def _tiled_wigner_source(source: str, real_type: str, reservoir_type: str,
                         batched: bool, graph: bool) -> str:
    """Fuse a Wigner increment into each final tiled cell write."""
    import re

    from polarism.noise.philox import cuda_source

    precision = "single" if real_type == "float" else "double"
    philox = cuda_source(precision).split('extern "C" __global__ void philox_words', 1)[0]
    start = source.index('extern "C" __global__ void tiled_step')
    header, body = source[start:].split(") {", 1)
    header += ", " + _wigner_arguments(real_type, graph)
    for suffix, variable in (("", "re"), ("+1", "im")):
        pattern = rf"(?m)^(\s*)psi_out\[2\*gid{re.escape(suffix)}\] = (.*);$"
        replacement = rf"\1{real_type} psi_final_{variable} = \2;"
        body, count = re.subn(pattern, replacement, body, count=1)
        if count != 1:
            raise ValueError("tiled psi final write not found")
    active_field = "nA" if reservoir_type == "double" else "nR"
    active = f"(({active_field}_new < 0) ? 0 : {active_field}_new)"
    last_field = "nR" if reservoir_type == "single" else "nI"
    anchor = re.search(rf"(?m)^.*{last_field}_out\[gid\] = \({last_field}_new < 0\) \? 0 : {last_field}_new;", body)
    if anchor is None:
        raise ValueError("tiled reservoir final write not found")
    increment = _wigner_increment(real_type, active,
                                  "member" if batched else "0", "gid", "gid",
                                  graph, tiled=True)
    body = body[:anchor.end()] + increment + body[anchor.end():]
    return philox + source[:start] + header + ") {" + body


def _without_reservoir(source: str, real_type: str) -> str:
    """Remove single-reservoir state and equations from a staged kernel."""
    import re

    pointer = re.compile(
        rf"^    (?:const )?{real_type}\* __restrict__ "
        r"(?:nR|nR0|k_nR|dnR_dt|nR_out|k1n|k2n),\n",
        re.MULTILINE,
    )
    source = pointer.sub("", source)
    source = re.sub(r"\bnR_val\b", f"({real_type})0", source)
    source = re.sub(r"\b(gpe_rhs|fused_stage_rhs|fused_final_stage)\b",
                    r"\1_none", source)
    source = re.sub(r"(?m)^    const (?:float|double) lap_nR = [^\n]*\n"
                    r"\s*\+ [^;]*;\n", "", source)
    source = re.sub(r"(?m)^    const (?:float|double) lap_nR = [^;]*;\n",
                    "", source)
    filtered = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if re.match(r"(?:const (?:float|double) (?:\(.*\)|nR_(?:up|dn|lt|rt)|k4n)|"
                    r"(?:float|double) n_new|dnR_dt\[|nR_out\[)", stripped):
            continue
        if "const " in stripped and f"({real_type})0 = nR" in stripped:
            continue
        if "nR0[tid] + c_dt * k_nR[tid]" in stripped:
            continue
        filtered.append(line)
    return "".join(filtered)


def _extend_cuda_source(source: str, real_type: str, reservoir_type: str,
                        batched: bool, terms, varying_potential: bool,
                        drive: bool) -> str:
    """Add local term, potential-envelope, and drive inputs to a flat kernel."""
    import re

    index = "potential_offset + spatial" if batched else "tid"
    cell = "spatial" if batched else "tid"
    active = {"none": f"({real_type})0", "single": "nR_val",
              "double": "nA_val", "quadratic-double": "nR_val"}[reservoir_type]
    inactive = ("nI_val" if reservoir_type in ("double", "quadratic-double")
                else f"({real_type})0")
    declarations = []
    if varying_potential:
        declarations.extend((f"const {real_type}* __restrict__ V_cap_re",
                             f"const {real_type}* __restrict__ V_cap_im"))
    if drive:
        declarations.extend((f"const {real_type}* __restrict__ drive_re",
                             f"const {real_type}* __restrict__ drive_im",
                             f"const {real_type}* __restrict__ drive_energy"))
    if terms:
        declarations.extend((f"const {real_type}* __restrict__ x_coords",
                             f"const {real_type}* __restrict__ y_coords"))
    declarations.extend((f"const {real_type}* __restrict__ feature_values",
                         "const int feature_index", "const int feature_members"))
    array_loads = []
    for term_number, term in enumerate(terms):
        for argument in term.cuda.arguments:
            ctype, name = argument[:2]
            layout = argument[2] if len(argument) == 3 else "raw"
            kernel_name = f"term_{term_number}_{name}"
            declaration_name = kernel_name + "_arg" if layout != "raw" else kernel_name
            declarations.append(f"{ctype.replace('{REAL}', real_type)} {declaration_name}")
            if layout != "raw":
                if "*" not in ctype:
                    raise ValueError("Shared or batched CUDA arguments must be arrays")
                value_type = ctype.replace("const", "").replace("*", "").replace(
                    "{REAL}", real_type).strip()
                position = "cell" if layout == "shared" else "tid"
                array_loads.append(f"    const {value_type} {kernel_name} = "
                                   f"{declaration_name}[{position}];\n")
    header, body = source.split(") {", 1)
    header += ", " + ", ".join(declarations)
    values_index = "(feature_index*feature_members+member)*3"
    locals_ = (
        f"    const int cell = {cell};\n"
        + ("    const int member = 0;\n" if not batched else "")
        + f"    const {real_type} potential_scale = feature_values[{values_index}];\n"
        + f"    const {real_type} drive_scale = feature_values[{values_index}+1];\n"
        + f"    const {real_type} t_stage = feature_values[{values_index}+2];\n"
    )
    if terms:
        locals_ += (f"    const {real_type} n_active = {active};\n"
                    f"    const {real_type} n_inactive = {inactive};\n"
                    f"    const {real_type} x = x_coords[cell];\n"
                    f"    const {real_type} y = y_coords[cell];\n")
        locals_ += "".join(array_loads)
    marker = f"    const {real_type} eff = "
    if marker not in body:
        raise ValueError("CUDA energy expression not found")
    body = body.replace(marker, locals_ + marker, 1)
    energy_terms = [term for term in enumerate(terms) if term[1].category == "energy"]
    rate_terms = [term for term in enumerate(terms) if term[1].category == "rate"]
    def expression(match, category):
        value = match.group(2)
        if category == "energy" and varying_potential:
            value = value.replace(f"V_re[{index}]",
                                  f"(V_re[{index}] * potential_scale + V_cap_re[{index}])")
        if category == "rate" and varying_potential:
            value = value.replace(f"v_im_scale * V_im[{index}] * inv_hbar",
                                  f"(V_im[{index}] * potential_scale + V_cap_im[{index}]) * inv_hbar")
        if category == "energy" and drive:
            value += " - drive_energy[tid]"
        chosen = energy_terms if category == "energy" else rate_terms
        for term_number, term in chosen:
            term_expr = term.cuda.expression
            for argument in term.cuda.arguments:
                name = argument[1]
                term_expr = re.sub(rf"\b{name}\b", f"term_{term_number}_{name}", term_expr)
            value += f" + ({term_expr})"
        return value
    for name, category in (("eff", "energy"), ("gl", "rate")):
        pattern = rf"(?m)(const {real_type} {name} = )([^;]+);"
        body, count = re.subn(pattern, lambda m: m.group(1) + expression(m, category) + ";",
                              body, count=1)
        if count != 1:
            raise ValueError(f"CUDA {name} expression not found")
    if drive:
        body = body.replace("relax * lap_re;",
                            "relax * lap_re + inv_hbar * drive_scale * drive_im[tid];")
        body = body.replace("relax * lap_im;",
                            "relax * lap_im - inv_hbar * drive_scale * drive_re[tid];")
    return header + ") {" + body


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
    batched: bool = False,
    physics_fields=(),
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
    energy_expression, rate_expression, relax_expression = _core_cuda_snippets(reservoir_type, real_type)
    fmt = dict(
        ENERGY_EXPRESSION=energy_expression,
        RATE_EXPRESSION=rate_expression,
        RELAX_EXPRESSION=relax_expression,
        RESERVOIR_RHS=_reservoir_cuda_snippet(reservoir_type, real_type),
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
    sources = tuple(s.replace("pump_scale * pump[tid]", f"{mul}(pump_scale, pump[tid])") for s in sources)
    if batched:
        return tuple(_batch_staged_source(s, real_type, bc_type, laplacian_type,
                                          physics_fields) for s in sources)
    return sources


def build_flat_1d_sources(bc_type, real_type="double", reservoir_type="single",
                          batched=False, physics_fields=()):
    """Generate flat one-dimensional kernels from the staged RK4 templates."""
    import re

    sources = build_kernel_source_1d(
        bc_type, real_type, reservoir_type, batched=batched,
        physics_fields=physics_fields,
    )
    result = []
    for source in sources:
        if batched:
            preamble = re.compile(
                r"    const int spatial = blockIdx.x.*?    const int j = spatial % nx;\n",
                re.DOTALL,
            )
            replacement = (
                "    const int spatial = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    const int N = nx;\n"
                "    if (spatial >= N) return;\n"
                "    const int member = blockIdx.y;\n"
                "    const int member_offset = member * N;\n"
                "    const int tid = member_offset + spatial;\n"
                "    const int potential_offset = potential_n == N ? 0 : member_offset;\n"
                "    const int pump_offset = pump_n == N ? 0 : member_offset;\n"
            )
            coordinate = "spatial"
        else:
            preamble = re.compile(
                r"    const int tid = blockIdx.x.*?    const int j = tid % nx;\n",
                re.DOTALL,
            )
            replacement = (
                "    const int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
                "    if (tid >= nx) return;\n"
            )
            coordinate = "tid"
        source, count = preamble.subn(replacement, source, count=1)
        if count != 1:
            raise ValueError("1D kernel preamble not found")
        neighbors = re.compile(r"    const int up = .*?    const int rt = .*?;\n", re.DOTALL)
        if bc_type == "periodic":
            replacement = (
                f"    const int lt = {coordinate} > 0 ? tid - 1 : tid + nx - 1;\n"
                f"    const int rt = {coordinate} < nx - 1 ? tid + 1 : tid - nx + 1;\n"
            )
        else:
            replacement = (
                f"    const int lt = {coordinate} > 0 ? tid - 1 : tid + 1;\n"
                f"    const int rt = {coordinate} < nx - 1 ? tid + 1 : tid - 1;\n"
            )
        source, count = neighbors.subn(replacement, source, count=1)
        if count != 1:
            raise ValueError("1D kernel neighbors not found")
        source = re.sub(
            r"    const (?:float|double) (?:psi|nR|nA|nI)_(?:up|dn)_(?:re|im) = [^;]*;\n",
            "", source,
        )
        source = re.sub(
            r"    const (?:float|double) (?:nR|nA|nI)_(?:up|dn) = [^;]*;\n",
            "", source,
        )
        source = re.sub(
            r"(const (?:float|double) lap_\w+ = )[^\n]*\* inv_dy2\n\s*\+ ([^;]*\* inv_dx2;)",
            r"\1\2", source,
        )
        body = source.split(") {", 1)[1]
        if re.search(r"\b(?:up|dn)\b|\binv_dy2\b", body):
            raise ValueError("1D kernel retains a y-axis access")
        result.append(source)
    return tuple(result)


def _batch_parameter_names(physics_fields):
    names = {"g_C": "g_C", "g_R": "g_R", "g_I": "g_I", "R": "R_cond",
             "gamma_C": "gamma_C", "gamma_R": "gamma_R", "gamma_I": "gamma_I",
             "gamma_A": "gamma_A", "R_IA": "R_IA", "R_AI": "R_AI",
             "kappa": "kappa", "kinetic_relaxation_eta": "kinetic_relaxation_eta",
             "reservoir_diffusion_R": "diff_R", "reservoir_diffusion_A": "diff_A",
             "reservoir_diffusion_I": "diff_I"}
    result = ["neg_hbar2_over_2m" for name in physics_fields
              if name in ("hbar", "m_eff")]
    if "hbar" in physics_fields:
        result.append("inv_hbar")
    result.extend(names[name] for name in physics_fields if name in names)
    return tuple(dict.fromkeys(result))


def _batch_staged_source(source, real_type, bc_type, laplacian_type,
                         physics_fields):
    import re

    head, body = source.split(") {", 1)
    head += (f", const int members, const int potential_n, const int pump_n, "
             f"const {real_type}* __restrict__ batch_constants")
    preamble = """
    const int spatial = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = nx * ny;
    if (spatial >= N) return;
    const int member = blockIdx.y;
    const int member_offset = member * N;
    const int tid = member_offset + spatial;
    const int potential_offset = potential_n == N ? 0 : member_offset;
    const int pump_offset = pump_n == N ? 0 : member_offset;
    const int i = spatial / nx;
    const int j = spatial % nx;
"""
    body = body.replace(PREAMBLE_1D, preamble)
    if bc_type == "periodic":
        body = body.replace(_PERIODIC_NEIGHBORS,
                            re.sub(r"(const int (?:up|dn|lt|rt) = )",
                                   r"\1member*N + ", _PERIODIC_NEIGHBORS))
    if laplacian_type == "isotropic-9pt":
        diag = _build_laplacian_code(laplacian_type, bc_type, real_type)[0]
        body = body.replace(diag, re.sub(r"(const int (?:ul|ur|dl|dr) = )",
                                         r"\1member*N + ", diag))
    body = body.replace("V_re[tid]", "V_re[potential_offset + spatial]")
    body = body.replace("V_im[tid]", "V_im[potential_offset + spatial]")
    body = body.replace("pump[tid]", "pump[pump_offset + spatial]")
    names = _batch_parameter_names(physics_fields)
    for name in names:
        body = re.sub(rf"\b{name}\b", f"{name}_member", body)
    loads = "".join(f"\n    const {real_type} {name}_member = "
                    f"__ldg(batch_constants + {i}*members+member);"
                    for i, name in enumerate(names))
    body = body.replace(preamble, preamble + loads, 1)
    return head + ") {" + body


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


def _tile_halo_indices_1d(nx: int, tile_x: int, block_x: int, bc_type: str):
    """Emulate the grid indices loaded by a one-dimensional tile."""
    import numpy as np

    return np.array([_tile_index(block_x * tile_x + i - 4, nx, bc_type)
                     for i in range(tile_x + 8)])


def _tile_shared_bytes_1d(tile_x: int, real_type: str,
                          reservoir_type: str) -> int:
    fields = 3 if reservoir_type == "single" else 4
    byte_width = 4 if real_type == "float" else 8
    return byte_width * fields * (2 * (tile_x + 8) + 3 * tile_x)


def _choose_tile_x(real_type: str, reservoir_type: str, block_limit: int,
                   sm_limit: int):
    candidates = (512, 256, 128, 64)
    for tile_x in candidates:
        needed = _tile_shared_bytes_1d(tile_x, real_type, reservoir_type)
        if needed <= block_limit and 2 * needed <= sm_limit:
            return tile_x
    for tile_x in candidates:
        if _tile_shared_bytes_1d(tile_x, real_type, reservoir_type) <= block_limit:
            return tile_x
    return None


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


def build_tiled_1d_kernel_source(bc_type: str, real_type: str,
                                 reservoir_type: str, tile_x: int,
                                 complex_potential: bool, graph: bool = False,
                                 batched: bool = False, physics_fields=()) -> str:
    """Build a whole-step one-dimensional kernel from the flat RHS."""
    import re

    rhs_source = build_flat_1d_sources(bc_type, real_type, reservoir_type)[0]
    fields = ("nR",) if reservoir_type == "single" else (
        ("nA", "nI") if reservoir_type == "double" else ("nR", "nI")
    )
    derivative_names = {"nR": "dnR_dt", "nA": "dnA_dt", "nI": "dnI_dt"}
    suffix = {"single": "", "double": "_double",
              "quadratic-double": "_qdouble"}[reservoir_type]
    header, body = rhs_source.split(") {", 1)
    body = body.rsplit("\n}", 1)[0]
    body, count = re.subn(
        r"    const int tid = blockIdx.x \* blockDim.x \+ threadIdx.x;\n"
        r"    if \(tid >= nx\) return;\n", "", body, count=1,
    )
    if count != 1:
        raise ValueError("1D tiled kernel preamble not found")
    neighbors = re.compile(r"    const int lt = .*?    const int rt = .*?;\n", re.DOTALL)
    if bc_type == "periodic":
        tile_neighbors = "    const int lt = tid - 1;\n    const int rt = tid + 1;\n"
    else:
        tile_neighbors = (
            "    const int x = tiled_coord_1d(vx, nx);\n"
            "    const int lt = tiled_coord_1d(vx - 1, nx) == (x > 0 ? x - 1 : 1)"
            " ? tid - 1 : tid + 1;\n"
            "    const int rt = tiled_coord_1d(vx + 1, nx) == (x < nx-1 ? x + 1 : nx-2)"
            " ? tid + 1 : tid - 1;\n"
        )
    body, count = neighbors.subn(tile_neighbors, body, count=1)
    if count != 1:
        raise ValueError("1D tiled kernel neighbors not found")
    spatial = "potential_offset + gid" if batched else "gid"
    pump_spatial = "pump_offset + gid" if batched else "gid"
    body = body.replace("V_re[tid]", f"V_re[{spatial}]")
    body = body.replace("V_im[tid]", f"V_im[{spatial}]")
    body = body.replace("pump[tid]", f"pump[{pump_spatial}]")
    if not complex_potential:
        body = body.replace(f"V_im[{spatial}]", f"V_re[{spatial}]")
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
        '__device__ __forceinline__ void tiled_rhs_1d(', header.strip(), count=1,
    )
    extra_indices = (" const int potential_offset, const int pump_offset,"
                     if batched else "")
    rhs_header = rhs_header.replace(
        "const int nx, const int ny,",
        f"const int nx, const int ny, const int tid, const int gid,"
        f" const int vx,{extra_indices}",
    )
    if bc_type == "periodic":
        coord_source = """__device__ __forceinline__ int tiled_coord_1d(int index, int size) {
    int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}"""
    else:
        coord_source = """__device__ __forceinline__ int tiled_coord_1d(int index, int size) {
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
    if batched:
        declarations += ["const int members", "const int potential_n",
                         "const int pump_n0", "const int pump_n1",
                         "const int pump_n2",
                         f"const {real_type}* __restrict__ batch_constants"]
    scale_values = ([f"pump_scales[pump_index + {i}]" for i in range(3)] if graph
                    else [f"scale{i}" for i in range(3)])
    count = 2 + len(fields)
    lines = [
        f"extern __shared__ {real_type} scratch[];",
        f"const int area = {tile_x}+8;",
        "const int lane = threadIdx.x;",
        f"const int base_x = blockIdx.x * {tile_x};",
    ]
    if batched:
        lines += [
            "const int member = blockIdx.y;",
            "const int member_offset = member * nx;",
            "const int potential_offset = potential_n == nx ? 0 : member_offset;",
            "const int pump_offset0 = pump_n0 == nx ? 0 : member_offset;",
            "const int pump_offset1 = pump_n1 == nx ? 0 : member_offset;",
            "const int pump_offset2 = pump_n2 == nx ? 0 : member_offset;",
        ]
        for name in ("psi_initial", *[f"{field}_initial" for field in fields],
                     "psi_out", *[f"{field}_out" for field in fields]):
            factor = "2*" if name.startswith("psi_") else ""
            lines.append(f"{name} += {factor}member_offset;")
        for index, name in enumerate(_batch_parameter_names(physics_fields)):
            lines.append(f"const {real_type} {name}_member = "
                         f"__ldg(batch_constants + {index}*members+member);")
    for bank in (0, 1):
        offset = bank * count
        lines.append(f"{real_type}* psi{bank} = scratch + {offset}*area;")
        for index, name in enumerate(fields):
            lines.append(f"{real_type}* {name}{bank} = scratch + {offset+2+index}*area;")
    k_base = 2 * count
    lines.append(f"{real_type}* kp = scratch + {k_base}*area;")
    for index, name in enumerate(fields):
        lines.append(f"{real_type}* k{name} = scratch + {k_base}*area + "
                     f"(2+{index})*{tile_x}*3;")
    lines += [
        "for (int p = lane; p < area; p += blockDim.x) {",
        "    const int gid = tiled_coord_1d(base_x + p - 4, nx);",
        "    psi0[2*p] = psi_initial[2*gid];",
        "    psi0[2*p+1] = psi_initial[2*gid+1];",
    ]
    for name in fields:
        lines.append(f"    {name}0[p] = {name}_initial[gid];")
    lines += ["}", "__syncthreads();", "#pragma unroll",
              "for (int stage = 0; stage < 4; ++stage) {",
              "    for (int p = lane; p < area; p += blockDim.x) {",
              "        if (p < stage+1 || p >= area-stage-1) continue;",
              "        const int vx = base_x + p - 4;",
              "        const int gid = tiled_coord_1d(vx, nx);",
              "        const int out = p - 4;",
              f"        {real_type} dp[2];"]
    for name in fields:
        lines.append(f"        {real_type} d{name}[1];")
    lines.append(f"        const {real_type}* input_psi = (stage & 1) ? psi1 : psi0;")
    for name in fields:
        lines.append(f"        const {real_type}* input_{name} = (stage & 1) ? {name}1 : {name}0;")
    lines += [
        f"        const {real_type}* stage_pump = stage == 0 ? pump0 : "
        "(stage == 3 ? pump2 : pump1);",
        f"        const {real_type} stage_scale = stage == 0 ? {scale_values[0]} : "
        f"(stage == 3 ? {scale_values[2]} : {scale_values[1]});",
    ]
    if batched:
        lines.append("        const int pump_offset = stage == 0 ? pump_offset0 : "
                     "(stage == 3 ? pump_offset2 : pump_offset1);")
    call_parameters = [f"{name}_member" if batched and name in
                       _batch_parameter_names(physics_fields) else name
                       for name in parameter_names]
    lines += [
        "        tiled_rhs_1d(input_psi, "
        + ", ".join(f"input_{name}" for name in fields)
        + ", V_re, V_im, stage_pump, dp, "
        + ", ".join(f"d{name}" for name in fields)
        + ", nx, ny, p, gid, vx, "
        + ("potential_offset, pump_offset, " if batched else "")
        + "stage_scale, v_im_scale, " + ", ".join(call_parameters) + ");",
        "        if (stage < 3) {",
        f"            const {real_type} c_dt = stage == 2 ? dt : half_dt;",
        f"            {real_type}* next_psi = (stage & 1) ? psi0 : psi1;",
        "            next_psi[2*p] = psi_initial[2*gid] + c_dt * dp[0];",
        "            next_psi[2*p+1] = psi_initial[2*gid+1] + c_dt * dp[1];",
    ]
    for name in fields:
        lines += [f"            {real_type}* next_{name} = (stage & 1) ? {name}0 : {name}1;",
                  f"            next_{name}[p] = {name}_initial[gid] + c_dt * d{name}[0];"]
    lines += ["        }", f"        if (p >= 4 && p < 4+{tile_x}) {{",
              "            if (stage < 3) {",
              f"                kp[stage*2*{tile_x}+2*out] = dp[0];",
              f"                kp[stage*2*{tile_x}+2*out+1] = dp[1];"]
    for name in fields:
        lines.append(f"                k{name}[stage*{tile_x}+out] = d{name}[0];")
    lines += ["            } else if (vx < nx) {",
              f"                psi_out[2*gid] = psi_initial[2*gid] + dt6 * (kp[2*out] + 2*kp[2*{tile_x}+2*out] + 2*kp[4*{tile_x}+2*out] + dp[0]);",
              f"                psi_out[2*gid+1] = psi_initial[2*gid+1] + dt6 * (kp[2*out+1] + 2*kp[2*{tile_x}+2*out+1] + 2*kp[4*{tile_x}+2*out+1] + dp[1]);"]
    for name in fields:
        lines += [f"                {real_type} {name}_new = {name}_initial[gid] + dt6 * (k{name}[out] + 2*k{name}[{tile_x}+out] + 2*k{name}[2*{tile_x}+out] + d{name}[0]);",
                  f"                {name}_out[gid] = ({name}_new < 0) ? 0 : {name}_new;"]
    lines += ["            }", "        }", "    }", "    __syncthreads();", "}"]
    kernel = "\n    ".join(lines)
    return (coord_source + "\n" + rhs_header + ") {\n" + body + "\n}\n"
            + f'extern "C" __global__ void tiled_step_1d{suffix}('
            + ",\n    ".join(declarations) + ") {\n    " + kernel + "\n}\n")


def build_tiled_kernel_source(bc_type: str, real_type: str, reservoir_type: str,
                              laplacian_type: str, tile_x: int, tile_y: int,
                              complex_potential: bool, graph: bool = False,
                              batched: bool = False, physics_fields=()) -> str:
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
    source = (coord_source + "\n"
            + rhs_header + ") {\n    const int i = gid / nx;\n    const int j = gid % nx;\n"
            + body + "\n}\n"
            + f'extern "C" __global__ void tiled_step{suffix}(' + ",\n    ".join(declarations)
            + ") {\n    " + kernel + "\n}\n")
    if batched:
        return _batch_tiled_source(source, real_type, physics_fields,
                                   parameter_names)
    return source


def _batch_tiled_source(source, real_type, physics_fields, parameter_names):
    import re

    device, kernel = source.split('extern "C" __global__ void tiled_step', 1)
    device = device.replace("const int nx, const int ny, const int tid, const int gid,",
                            "const int nx, const int ny, const int tid, const int spatial, "
                            "const int potential_offset, const int pump_offset,")
    device = device.replace("const int i = gid / nx;", "const int i = spatial / nx;")
    device = device.replace("const int j = gid % nx;", "const int j = spatial % nx;")
    device = device.replace("V_re[gid]", "V_re[potential_offset + spatial]")
    device = device.replace("V_im[gid]", "V_im[potential_offset + spatial]")
    device = device.replace("pump[gid]", "pump[pump_offset + spatial]")
    marker = ") {\n    "
    header, body = kernel.split(marker, 1)
    state_names = tuple(dict.fromkeys(re.findall(r"\b\w+_(?:initial|out)\b", header)))
    for name in state_names:
        header = re.sub(rf"\b{name}\b", f"{name}_batch", header)
    header += (f",\n    const int members, const int potential_n, "
               f"const int pump_n0, const int pump_n1, const int pump_n2, "
               f"const {real_type}* __restrict__ batch_constants")
    body = body.replace("const int lane = threadIdx.x;",
                        "const int lane = threadIdx.x;\n    const int member = blockIdx.z;\n"
                        "    const int spatial_n = nx * ny;\n"
                        "    const int member_offset = member * spatial_n;\n"
                        "    const int potential_offset = potential_n == spatial_n ? 0 : member_offset;\n"
                        "    const int pump_offset0 = pump_n0 == spatial_n ? 0 : member_offset;\n"
                        "    const int pump_offset1 = pump_n1 == spatial_n ? 0 : member_offset;\n"
                        "    const int pump_offset2 = pump_n2 == spatial_n ? 0 : member_offset;", 1)
    body = body.replace("const int gid = tiled_coord(vy, ny)*nx + tiled_coord(vx, nx);",
                        "const int spatial = tiled_coord(vy, ny)*nx + tiled_coord(vx, nx);\n"
                        "        const int gid = spatial;")
    names = _batch_parameter_names(physics_fields)
    pointers = "".join(
        f"\n    {'const ' if name.endswith('_initial') else ''}{real_type}* "
        f"__restrict__ {name} = {name}_batch + "
        f"{'2*' if name.startswith('psi_') else ''}member_offset;"
        for name in state_names
    )
    loads = "".join(f"\n    const {real_type} {name}_member = "
                    f"__ldg(batch_constants + {i}*members+member);"
                    for i, name in enumerate(names))
    body = body.replace("const int member_offset = member * spatial_n;",
                        "const int member_offset = member * spatial_n;" + pointers + loads, 1)
    call = ", ".join(parameter_names) + ");"
    member_call = ", ".join(f"{name}_member" if name in names else name
                            for name in parameter_names) + ");"
    body = body.replace(call, member_call)
    body = body.replace("nx, ny, p, gid, vx, vy, pitch, stage_scale,",
                        "nx, ny, p, spatial, potential_offset, stage_pump_offset, "
                        "vx, vy, pitch, stage_scale,")
    body = body.replace("const int out = (y-4)*",
                        "const int stage_pump_offset = stage == 0 ? pump_offset0 : "
                        "(stage == 3 ? pump_offset2 : pump_offset1);\n"
                        "        const int out = (y-4)*", 1)
    return device + 'extern "C" __global__ void tiled_step' + header + marker + body


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


def build_spinor_tiled_rhs_source(real_type: str, boundary: str,
                                  tile_x: int, tile_y: int, terms=()) -> str:
    """Build a two-field shared-memory stencil with corner halo samples."""
    if real_type not in ("float", "double") or boundary not in ("periodic", "closed-interval"):
        raise ValueError("Invalid spinor CUDA stencil variant")
    periodic = 1 if boundary == "periodic" else 0
    source = f'''typedef {real_type} real;
__device__ __forceinline__ int spinor_coord(int value, int length) {{
    if ({periodic}) return (value % length + length) % length;
    if (length == 1) return 0;
    while (value < 0 || value >= length) value = value < 0 ? -value : 2*length-2-value;
    return value;
}}
extern "C" __global__ void spinor_tiled_rhs(
    const real* field, const real* active, const real* inactive,
    const real* potential_re, const real* potential_im,
    const real* constants, real* output, int nx, int ny,
    real inv_dx2, real inv_dy2, real inv_4dxdy) {{
    const int tx = {tile_x}, ty = {tile_y}, pitch = tx+2, area = (tx+2)*(ty+2);
    const int spatial_n = nx*ny, member = blockIdx.z, lane = threadIdx.x;
    const int base_x = blockIdx.x*tx, base_y = blockIdx.y*ty;
    extern __shared__ real scratch[];
    for (int p = lane; p < area; p += blockDim.x) {{
        int x = spinor_coord(base_x+p%pitch-1, nx);
        int y = spinor_coord(base_y+p/pitch-1, ny);
        int cell = y*nx+x;
        for (int c = 0; c < 2; ++c) {{
            int source = (member*2+c)*spatial_n+cell;
            scratch[(2*c)*area+p] = field[2*source];
            scratch[(2*c+1)*area+p] = field[2*source+1];
        }}
    }}
    __syncthreads();
    if (lane >= tx*ty) return;
    int local_x = lane%tx, local_y = lane/tx;
    int gx = base_x+local_x, gy = base_y+local_y;
    if (gx >= nx || gy >= ny) return;
    int cell = gy*nx+gx, p = (local_y+1)*pitch+local_x+1;
    const real* q = constants+11*member;
    real hbar = q[0], mass = q[1], gC = q[2], alpha2 = q[3];
    real zeeman = q[4], beta = q[5], gR = q[6], gI = q[7];
    real R = q[8], gammaC = q[9], eta = q[10];
    real kinetic = -hbar*hbar/((real)2*mass), relax = eta*hbar/((real)2*mass);
    real inv_hbar = (real)1/hbar;
    real densities[2];
    for (int c = 0; c < 2; ++c) {{
        real re = scratch[(2*c)*area+p], im = scratch[(2*c+1)*area+p];
        densities[c] = re*re+im*im;
    }}
    real lap[2][2], dif[2][2], crs[2][2];
    for (int c = 0; c < 2; ++c) {{
        for (int part = 0; part < 2; ++part) {{
            const int plane = (2*c+part)*area;
            real centre = scratch[plane+p];
            real sx = scratch[plane+p+1]+scratch[plane+p-1]-2*centre;
            real second_x = sx*inv_dx2;
            lap[c][part] = second_x;
            dif[c][part] = second_x;
            crs[c][part] = 0;
            if (ny > 1) {{
                real sy = scratch[plane+p+pitch]+scratch[plane+p-pitch]-2*centre;
                lap[c][part] += sy*inv_dy2;
                dif[c][part] -= sy*inv_dy2;
                crs[c][part] = (scratch[plane+p+pitch+1]-scratch[plane+p+pitch-1]
                              -scratch[plane+p-pitch+1]+scratch[plane+p-pitch-1])*inv_4dxdy;
            }}
        }}
    }}
    for (int c = 0; c < 2; ++c) {{
        int index = (member*2+c)*spatial_n+cell, other = 1-c;
        real re = scratch[(2*c)*area+p], im = scratch[(2*c+1)*area+p];
        real lap_re = lap[c][0], lap_im = lap[c][1];
        real diff_re = dif[other][0], diff_im = dif[other][1];
        real cross_re = crs[other][0], cross_im = crs[other][1];
        real sign = c == 0 ? (real)1 : (real)-1;
        real te_re = -beta*(diff_re+2*sign*cross_im);
        real te_im = -beta*(diff_im-2*sign*cross_re);
        real nr = active[index], ni = inactive[index];
        real energy = potential_re[index]+gC*densities[c]+alpha2*densities[other]
                    +sign*zeeman*(real)0.5+gR*nr+gI*ni;
        real gain = (R*nr-gammaC)*(real)0.5+potential_im[index]*inv_hbar;
        output[2*index] = gain*re+(energy*im+kinetic*lap_im+te_im)*inv_hbar+relax*nr*lap_re;
        output[2*index+1] = gain*im-(energy*re+kinetic*lap_re+te_re)*inv_hbar+relax*nr*lap_im;
    }}
}}
'''
    if not terms:
        return source
    import re

    declarations = ("const real* x_coords", "const real* y_coords", "real t_stage")
    loads = []
    energy_terms = []
    rate_terms = []
    for number, term in enumerate(terms):
        expression = term.cuda.expression
        for ctype, name, *layout in term.cuda.arguments:
            kind = layout[0] if layout else "raw"
            kernel_name = f"term_{number}_{name}"
            declared_name = kernel_name if kind == "raw" else f"{kernel_name}_arg"
            declarations += (f"{ctype.replace('{REAL}', real_type)} {declared_name}",)
            if kind != "raw":
                if "*" not in ctype:
                    raise ValueError("Shared or batched CUDA arguments must be arrays")
                value_type = ctype.replace("const", "").replace("*", "").replace(
                    "{REAL}", real_type).strip()
                offset = "cell" if kind == "shared" else "member*spatial_n+cell"
                loads.append(f"        const {value_type} {kernel_name} = "
                             f"{declared_name}[{offset}];")
            expression = re.sub(rf"\b{name}\b", kernel_name, expression)
        if term.category == "energy":
            energy_terms.append(f"({expression})")
        else:
            rate_terms.append(f"({expression})")
    source = source.replace("real inv_dx2, real inv_dy2, real inv_4dxdy) {",
                            "real inv_dx2, real inv_dy2, real inv_4dxdy, "
                            + ", ".join(declarations) + ") {")
    local = ("        real rho = densities[c], n_active = nr, n_inactive = ni;\n"
             "        real psi_re = re, psi_im = im;\n"
             "        real x = x_coords[cell], y = y_coords[cell];\n")
    if loads:
        local += "\n".join(loads) + "\n"
    local += ("        real extra_energy = " + (" + ".join(energy_terms) or "(real)0") + ";\n"
              "        real extra_rate = " + (" + ".join(rate_terms) or "(real)0") + ";\n")
    source = source.replace("        real energy = potential_re[index]", local +
                            "        real energy = potential_re[index]")
    source = source.replace("+sign*zeeman*(real)0.5+gR*nr+gI*ni;",
                            "+sign*zeeman*(real)0.5+gR*nr+gI*ni+extra_energy;")
    source = source.replace("+potential_im[index]*inv_hbar;",
                            "+potential_im[index]*inv_hbar+extra_rate;")
    return source


def build_spinor_tiled_stage_source(real_type, boundary, tile_x, tile_y,
                                     terms=(), graph=False,
                                     varying_potential=False, drive=False):
    """Build one paired field and reservoir RK derivative stage."""
    source = build_spinor_tiled_rhs_source(real_type, boundary, tile_x, tile_y, terms)
    source = source.replace("void spinor_tiled_rhs(", "void spinor_tiled_stage(")
    source = source.replace(
        "const real* field, const real* active, const real* inactive,",
        "const real* field0, const real* previous_field,\n"
        "    const real* active0, const real* inactive0,\n"
        "    const real* previous_active, const real* previous_inactive,")
    scale_args = ("const real* scales, int scale_index" if graph else "real scale")
    scale = "scales[scale_index]" if graph else "scale"
    source = source.replace(
        "real inv_dx2, real inv_dy2, real inv_4dxdy",
        "real inv_dx2, real inv_dy2, real inv_4dxdy,\n"
        "    const real* pump, int pump_n, const real* weights, int res_components,\n"
        f"    int reservoir_model, real factor, {scale_args},\n"
        "    real* derivative_active, real* derivative_inactive")
    source = source.replace("constants+11*member", "constants+17*member")
    source = source.replace(
        "scratch[(2*c)*area+p] = field[2*source];\n"
        "            scratch[(2*c+1)*area+p] = field[2*source+1];",
        "scratch[(2*c)*area+p] = field0[2*source]+factor*previous_field[2*source];\n"
        "            scratch[(2*c+1)*area+p] = field0[2*source+1]+factor*previous_field[2*source+1];")
    source = source.replace(
        "real nr = active[index], ni = inactive[index];",
        "int rindex = (member*res_components+(res_components == 1 ? 0 : c))*spatial_n+cell;\n"
        "        real nr = reservoir_model == 3 ? (real)0 : active0[rindex]+factor*previous_active[rindex];\n"
        "        real ni = (reservoir_model == 0 || reservoir_model == 3) ? (real)0 :\n"
        "            inactive0[rindex]+factor*previous_inactive[rindex];")
    anchor = "        output[2*index+1] = gain*im-(energy*re+kinetic*lap_re+te_re)*inv_hbar+relax*nr*lap_im;"
    reservoir = f'''
        if (reservoir_model != 3 && (res_components == 2 || c == 0)) {{
            real density = res_components == 1 ? densities[0]+densities[1] : densities[c];
            real weight = res_components == 1 ? (real)1 : weights[2*member+c];
            real source = {scale}*pump[pump_n == spatial_n ? cell : member*spatial_n+cell]*weight;
            real gammaR=q[11], gammaI=q[12], gammaA=q[13];
            real ria=q[14], rai=q[15], kappa=q[16];
            if (reservoir_model == 0) {{
                derivative_active[rindex] = source-(gammaR+R*density)*nr;
            }} else if (reservoir_model == 1) {{
                derivative_active[rindex] = ria*ni-(gammaA+rai+R*density)*nr;
                derivative_inactive[rindex] = source-(gammaI+ria)*ni+rai*nr;
            }} else {{
                real transfer=kappa*ni*ni;
                derivative_active[rindex] = transfer-(gammaR+R*density)*nr;
                derivative_inactive[rindex] = source-transfer-gammaI*ni;
            }}
        }}'''
    source = source.replace(anchor, anchor + reservoir)
    if varying_potential or drive:
        feature_args = []
        if varying_potential:
            feature_args.extend(("const real* cap_re", "const real* cap_im"))
        if drive:
            feature_args.extend(("const real* drive_re", "const real* drive_im",
                                 "const real* drive_energy"))
        feature_args.extend(("const real* feature_values", "int feature_index",
                             "int feature_members"))
        source = source.replace(
            "real* derivative_active, real* derivative_inactive",
            "real* derivative_active, real* derivative_inactive,\n    " +
            ", ".join(feature_args))
        source = source.replace("const real* q = constants+17*member;",
            "const real* q = constants+17*member;\n"
            "    int feature_offset=(feature_index*feature_members+member)*3;\n"
            "    real potential_scale=feature_values[feature_offset];\n"
            "    real drive_scale=feature_values[feature_offset+1];")
        if varying_potential:
            source = source.replace("potential_re[index]+gC",
                "(potential_re[index]*potential_scale+cap_re[index])+gC")
            source = source.replace("+potential_im[index]*inv_hbar",
                "+(potential_im[index]*potential_scale+cap_im[index])*inv_hbar")
        if drive:
            if varying_potential:
                source = source.replace(
                    "(potential_re[index]*potential_scale+cap_re[index])+gC",
                    "(potential_re[index]*potential_scale+cap_re[index]-drive_energy[index])+gC")
            else:
                source = source.replace("potential_re[index]+gC",
                    "(potential_re[index]-drive_energy[index])+gC")
            source = source.replace("+relax*nr*lap_re;",
                "+relax*nr*lap_re+drive_scale*drive_im[index]*inv_hbar;")
            source = source.replace("+relax*nr*lap_im;",
                "+relax*nr*lap_im-drive_scale*drive_re[index]*inv_hbar;")
    if graph:
        source = source.replace("real t_stage", "const real* stage_times, int stage_index")
        source = source.replace("t_stage", "stage_times[stage_index]")
    return source


def build_spinor_finish_source(real_type):
    """Combine four paired field and reservoir derivatives."""
    return f'''typedef {real_type} real;
extern "C" __global__ void spinor_finish(
    const real* field0, const real* k0, const real* k1, const real* k2, const real* k3,
    const real* active0, const real* a0, const real* a1, const real* a2, const real* a3,
    const real* inactive0, const real* b0, const real* b1, const real* b2, const real* b3,
    real* field_out, real* active_out, real* inactive_out,
    int field_n, int reservoir_n, int reservoir_model, real sixth) {{
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if (j < 2*field_n) {{
        field_out[j]=field0[j]+sixth*(k0[j]+(real)2*k1[j]+(real)2*k2[j]+k3[j]);
    }}
    if (j < reservoir_n && reservoir_model != 3) {{
        active_out[j]=fmax((real)0,active0[j]+sixth*(a0[j]+(real)2*a1[j]+(real)2*a2[j]+a3[j]));
        if (reservoir_model != 0) {{
            inactive_out[j]=fmax((real)0,inactive0[j]+sixth*(b0[j]+(real)2*b1[j]+(real)2*b2[j]+b3[j]));
        }}
    }}
}}
'''


_SPINOR_RESERVOIR_MODELS = {"SingleReservoir": 0, "DoubleReservoir": 1,
                            "QuadraticDoubleReservoir": 2, "NoReservoir": 3}
_SPINOR_STEP_RESERVOIR_PLANES = {0: 1, 1: 2, 2: 2, 3: 0}
_SPINOR_STEP_STAGES = {"whole": (0, 4), "first": (0, 2), "second": (2, 4)}


def spinor_step_planes(reservoir_model: int, res_components: int) -> int:
    """Count the shared-memory planes of one spinor state bank."""
    return 4 + _SPINOR_STEP_RESERVOIR_PLANES[reservoir_model] * (
        res_components if reservoir_model != 3 else 0)


def _spinor_step_bytes(tile_x, tile_y, two_d, planes, real_type, halo):
    rows = tile_y + (2 * halo if two_d else 0)
    return 2 * planes * (tile_x + 2 * halo) * rows * (4 if real_type == "float" else 8)


def choose_spinor_step_tile(real_type, two_d, planes, block_limit, sm_limit, halo):
    """Pick the largest spinor step tile that fits shared memory."""
    candidates = ((32, 16), (32, 8), (16, 16), (16, 8), (8, 8)) if two_d else (
        (256, 1), (128, 1), (64, 1))
    for tile in candidates:
        needed = _spinor_step_bytes(*tile, two_d, planes, real_type, halo)
        if needed <= block_limit and 2 * needed <= sm_limit:
            return tile
    for tile in candidates:
        if _spinor_step_bytes(*tile, two_d, planes, real_type, halo) <= block_limit:
            return tile
    return None


def build_spinor_step_source(real_type, tile_x, tile_y, two_d, reservoir_model,
                             res_components, terms=(), graph=False,
                             varying_potential=False, drive=False, phase="whole",
                             relaxation=True):
    """Build a periodic spinor RK4 kernel covering stages of one step.

    ``whole`` runs all four stages with a four-cell tile halo. ``first`` runs
    stages 0-1 and stores the stage-1 derivative with the running weighted sum;
    ``second`` resumes from those arrays and runs stages 2-3.

    Every stage before the last writes its staged values to shared memory,
    including the unread final store of ``first``. Guarding that store with
    the stage index made NVRTC 12.9 emit FP64 code that read unwritten shared
    memory when a thread covers more than one tile cell.
    """
    import re

    if real_type not in ("float", "double") or phase not in _SPINOR_STEP_STAGES:
        raise ValueError("Invalid spinor CUDA step variant")
    begin, end = _SPINOR_STEP_STAGES[phase]
    halo = end - begin
    model = reservoir_model
    comps = res_components if model != 3 else 0
    has_inactive = _SPINOR_STEP_RESERVOIR_PLANES[model] == 2
    planes = spinor_step_planes(model, res_components)
    halo_y = halo if two_d else 0
    rows = tile_y + 2 * halo_y
    block = 256
    iterations = ((tile_x + 2 * halo) * rows + block - 1) // block

    def cell(plane, offset=""):
        return f"bank[{plane}+p{offset}]"

    def lap_expression(component, part, sign):
        text = f"inv_dx2*sums_x[{component}][{part}]"
        if two_d:
            text += f"{sign}inv_dy2*sums_y[{component}][{part}]"
        return text

    if relaxation:
        operator_definitions = (
            f"                const real lap_re = {lap_expression('c', 0, '+')};\n"
            f"                const real lap_im = {lap_expression('c', 1, '+')};\n"
            f"                const real diff_re = {lap_expression('other', 0, '-')};\n"
            f"                const real diff_im = {lap_expression('other', 1, '-')};\n")
        relaxed_definition = "                const real relaxed = relax*nr;\n"
        operator_re = "kinetic_h*lap_im+relaxed*lap_re-beta_h*diff_im"
        operator_im = "-kinetic_h*lap_re+relaxed*lap_im+beta_h*diff_re"
    else:
        operator_definitions = relaxed_definition = ""
        operator_re = ("kinetic_x*sums_x[c][1]-beta_x*sums_x[other][1]" + (
            "+kinetic_y*sums_y[c][1]+beta_y*sums_y[other][1]" if two_d else ""))
        operator_im = ("-kinetic_x*sums_x[c][0]+beta_x*sums_x[other][0]" + (
            "-kinetic_y*sums_y[c][0]-beta_y*sums_y[other][0]" if two_d else ""))
    cross_definition = ("                const real cross_term = "
                        "c == 0 ? cross_coupling : -cross_coupling;\n" if two_d else "")
    cross_re = "+cross_term*crs[other][0]" if two_d else ""
    cross_im = "+cross_term*crs[other][1]" if two_d else ""
    gr_term = "" if model == 3 else "+gR*nr"
    gi_term = "+gI*ni" if has_inactive else ""
    energy_terms = []
    rate_terms = []
    loads = []
    declarations = []
    if varying_potential:
        declarations.extend(("const real* cap_re", "const real* cap_im"))
    if drive:
        declarations.extend(("const real* drive_re", "const real* drive_im",
                             "const real* drive_energy"))
    if varying_potential or drive:
        declarations.extend(("const real* feature_values", "int feature_index",
                             "int feature_members"))
    if terms:
        declarations.extend(("const real* x_coords", "const real* y_coords"))
        declarations.append("const real* stage_times, int stage_index" if graph
                            else "real time0, real time1, real time2")
    for number, term in enumerate(terms):
        expression = term.cuda.expression
        for ctype, name, *layout in term.cuda.arguments:
            kind = layout[0] if layout else "raw"
            kernel_name = f"term_{number}_{name}"
            concrete = ctype.replace("{REAL}", real_type)
            if kind == "raw":
                declarations.append(f"{concrete} {kernel_name}")
            else:
                if "*" not in ctype:
                    raise ValueError("Shared or batched CUDA arguments must be arrays")
                declarations.append(f"{concrete} {kernel_name}_arg")
                value_type = concrete.replace("const", "").replace("*", "").strip()
                offset = "cell" if kind == "shared" else "member*spatial_n+cell"
                loads.append(f"                const {value_type} {kernel_name} = "
                             f"{kernel_name}_arg[{offset}];")
            expression = re.sub(rf"\b{name}\b", kernel_name, expression)
        (energy_terms if term.category == "energy" else rate_terms).append(
            f"({expression})")
    scale_args = ("const real* scales, int scale_index" if graph else
                  "real scale0, real scale1, real scale2")
    scale_values = ("scales[scale_index+stage_slot]" if graph else
                    "(stage_slot == 0 ? scale0 : (stage_slot == 1 ? scale1 : scale2))")
    partial = ("real* __restrict__ {0}_field, real* __restrict__ {0}_active, "
               "real* __restrict__ {0}_inactive")
    partial_in = partial.replace("real* __restrict__", "const real* __restrict__")
    outputs = {"whole": ["real* __restrict__ field_out", "real* __restrict__ active_out",
                         "real* __restrict__ inactive_out"],
               "first": [partial.format("k"), partial.format("acc")],
               "second": ["real* __restrict__ field_out", "real* __restrict__ active_out",
                          "real* __restrict__ inactive_out", partial_in.format("k"),
                          partial_in.format("acc")]}[phase]
    signature = ["const real* __restrict__ field0", "const real* __restrict__ active0",
                 "const real* __restrict__ inactive0",
                 "const real* __restrict__ potential_re",
                 "const real* __restrict__ potential_im",
                 "const real* __restrict__ constants", "const real* __restrict__ pump",
                 "int pump_n", "const real* __restrict__ weights", *outputs,
                 "int nx", "int ny", "real inv_dx2", "real inv_dy2", "real inv_4dxdy",
                 "real half_dt", "real dt", "real dt6", scale_args, *declarations]
    feature_setup = ""
    if varying_potential or drive:
        feature_setup = (
            "        const int feature_offset="
            "((feature_index+stage_slot)*feature_members+member)*3;\n"
            "        const real potential_scale=feature_values[feature_offset];\n"
            "        const real drive_scale=feature_values[feature_offset+1];\n")
    term_time = ""
    if terms:
        term_time = ("        const real t_stage="
                     + ("stage_times[stage_index+stage_slot];\n" if graph else
                        "(stage_slot == 0 ? time0 : (stage_slot == 1 ? time1 : time2));\n"))
    potential_re = "potential_re[index]"
    potential_im = "potential_im[index]"
    if varying_potential:
        potential_re = "(potential_re[index]*potential_scale+cap_re[index])"
        potential_im = "(potential_im[index]*potential_scale+cap_im[index])"
    if drive:
        potential_re = f"({potential_re}-drive_energy[index])"
    drive_re = "+drive_scale*drive_im[index]*inv_hbar" if drive else ""
    drive_im = "-drive_scale*drive_re[index]*inv_hbar" if drive else ""
    extra = ""
    extra_energy = extra_rate = ""
    if terms:
        extra = ("                const real rho = densities[c], n_active = nr, n_inactive = ni;\n"
                 "                const real psi_re = re, psi_im = im;\n"
                 "                const real x = x_coords[cell], y = y_coords[cell];\n")
        if loads:
            extra += "\n".join(loads) + "\n"
        extra += ("                const real extra_energy = " +
                  (" + ".join(energy_terms) or "(real)0") + ";\n"
                  "                const real extra_rate = " +
                  (" + ".join(rate_terms) or "(real)0") + ";\n")
        extra_energy = "+extra_energy"
        extra_rate = "+extra_rate"
    reservoir_block = ""
    if model != 3:
        reservoir_derivative = {
            0: "d[4+rc] = source-(gammaR+R*density)*nr;",
            1: ("d[4+rc] = ria*ni-(gammaA+rai+R*density)*nr;\n"
                f"                    d[4+{comps}+rc] = source-(gammaI+ria)*ni+rai*nr;"),
            2: ("const real transfer = kappa*ni*ni;\n"
                "                    d[4+rc] = transfer-(gammaR+R*density)*nr;\n"
                f"                    d[4+{comps}+rc] = source-transfer-gammaI*ni;"),
        }[model]
        reservoir_block = f'''
                if ({comps} == 2 || c == 0) {{
                    const real density = {comps} == 1 ? densities[0]+densities[1] : densities[c];
                    const real weight = {comps} == 1 ? (real)1 : weights[2*member+c];
                    const real source = stage_scale*pump_value[it]*weight;
                    {reservoir_derivative}
                }}'''
    nr_load = "(real)0" if model == 3 else "bank[(4+rc)*area+p]"
    ni_load = "(real)0" if not has_inactive else f"bank[(4+{comps}+rc)*area+p]"
    final_value = ("init[it][{k}]+dt6*(acc[it][{k}]+d[{k}])")

    def field_loop(body):
        return ("                    for (int c = 0; c < 2; ++c) {\n"
                "                        const int target = (member*2+c)*spatial_n+cell;\n"
                + body + "                    }\n")

    def reservoir_loop(body):
        if model == 3:
            return ""
        return (f"                    for (int r = 0; r < {comps}; ++r) {{\n"
                f"                        const int rindex = (member*{comps}+r)*spatial_n+cell;\n"
                + body + "                    }\n")

    def write_all(field_name, active_name, inactive_name, value):
        text = field_loop(
            f"                        {field_name}[2*target] = {value.format(k='2*c')};\n"
            f"                        {field_name}[2*target+1] = {value.format(k='2*c+1')};\n")
        text += reservoir_loop(
            f"                        {active_name}[rindex] = {value.format(k='4+r')};\n" + (
                f"                        {inactive_name}[rindex] = "
                f"{value.format(k=f'4+{comps}+r')};\n" if has_inactive else ""))
        return text

    clamped = "fmax((real)0, " + final_value + ")"
    final_write = (field_loop(
        "                        field_out[2*target] = " + final_value.format(k="2*c") + ";\n"
        "                        field_out[2*target+1] = " + final_value.format(k="2*c+1") + ";\n"
        ) + reservoir_loop(
        "                        active_out[rindex] = " + clamped.format(k="4+r") + ";\n" + (
            "                        inactive_out[rindex] = " +
            clamped.format(k=f"4+{comps}+r") + ";\n" if has_inactive else "")))
    partial_write = (write_all("k_field", "k_active", "k_inactive", "d[{k}]")
                     + write_all("acc_field", "acc_active", "acc_inactive", "acc[it][{k}]"))
    load_extra = ""
    if phase == "second":
        load_extra = f'''
            real kin[{planes}];
            for (int c = 0; c < 2; ++c) {{
                const int source = (member*2+c)*spatial_n+cell;
                kin[2*c] = k_field[2*source];
                kin[2*c+1] = k_field[2*source+1];
            }}''' + ("" if model == 3 else f'''
            for (int r = 0; r < {comps}; ++r) {{
                const int rindex = (member*{comps}+r)*spatial_n+cell;
                kin[4+r] = k_active[rindex];''' + (f'''
                kin[4+{comps}+r] = k_inactive[rindex];''' if has_inactive else "") + '''
            }''') + f'''
            for (int k = 0; k < {planes}; ++k) staged[k] = init[it][k]+half_dt*kin[k];
            if (px >= {halo} && px < {halo}+tx && py >= {halo_y} && py < {halo_y}+{tile_y}) {{
                for (int c = 0; c < 2; ++c) {{
                    const int source = (member*2+c)*spatial_n+cell;
                    acc[it][2*c] = acc_field[2*source];
                    acc[it][2*c+1] = acc_field[2*source+1];
                }}''' + ("" if model == 3 else f'''
                for (int r = 0; r < {comps}; ++r) {{
                    const int rindex = (member*{comps}+r)*spatial_n+cell;
                    acc[it][4+r] = acc_active[rindex];''' + (f'''
                    acc[it][4+{comps}+r] = acc_inactive[rindex];''' if has_inactive else "") + '''
                }''') + '''
            }'''
    else:
        load_extra = f'''
            for (int k = 0; k < {planes}; ++k) staged[k] = init[it][k];'''
    load_reservoir = ""
    if model != 3:
        load_reservoir = f'''
            for (int r = 0; r < {comps}; ++r) {{
                const int rindex = (member*{comps}+r)*spatial_n+cell;
                init[it][4+r] = active0[rindex];'''
        if has_inactive:
            load_reservoir += f'''
                init[it][4+{comps}+r] = inactive0[rindex];'''
        load_reservoir += "\n            }"
    last = end - 1
    stage_end = "write_partial" if phase == "first" else "next_input"
    source = f'''typedef {real_type} real;
extern "C" __global__ void __launch_bounds__(256, 2) spinor_tiled_step(
    {(","+chr(10)+"    ").join(signature)}) {{
    const int tx = {tile_x}, pitch = tx+{2*halo}, rows = {rows}, area = pitch*rows;
    const int spatial_n = nx*ny, member = blockIdx.z, lane = threadIdx.x;
    const int base_x = blockIdx.x*tx, base_y = blockIdx.y*{tile_y};
    extern __shared__ real scratch[];
    const real* q = constants+17*member;
    const real hbar = q[0], mass = q[1], gC = q[2], alpha2 = q[3];
    const real zeeman = q[4], beta = q[5], gR = q[6], gI = q[7];
    const real R = q[8], gammaC = q[9], eta = q[10];
    const real gammaR = q[11], gammaI = q[12], gammaA = q[13];
    const real ria = q[14], rai = q[15], kappa = q[16];
    const real relax = eta*hbar/((real)2*mass);
    const real inv_hbar = (real)1/hbar;
    const real kinetic_h = -hbar/((real)2*mass), beta_h = beta*inv_hbar;
    const real cross_coupling = (real)2*beta_h*inv_4dxdy;
    const real half_R = R*(real)0.5, half_gammaC = gammaC*(real)0.5;
    const real zeeman_half = zeeman*(real)0.5;
    const real kinetic_x = kinetic_h*inv_dx2, kinetic_y = kinetic_h*inv_dy2;
    const real beta_x = beta_h*inv_dx2, beta_y = beta_h*inv_dy2;
    real init[{iterations}][{planes}], acc[{iterations}][{planes}];
    int cells[{iterations}];
    real pump_value[{iterations}];
    #pragma unroll
    for (int it = 0; it < {iterations}; ++it) {{
        const int p = lane+it*{block};
        if (p < area) {{
            const int px = p%pitch, py = p/pitch;
            const int x = (base_x+px-{halo}+nx*8)%nx;
            const int y = {f"(base_y+py-{halo_y}+ny*8)%ny" if two_d else "0"};
            const int cell = y*nx+x;
            real staged[{planes}];
            cells[it] = cell;
            pump_value[it] = pump[pump_n == spatial_n ? cell : member*spatial_n+cell];
            for (int c = 0; c < 2; ++c) {{
                const int source = (member*2+c)*spatial_n+cell;
                init[it][2*c] = field0[2*source];
                init[it][2*c+1] = field0[2*source+1];
            }}{load_reservoir}{load_extra}
            for (int k = 0; k < {planes}; ++k) scratch[k*area+p] = staged[k];
        }}
    }}
    __syncthreads();
    #pragma unroll 1
    for (int stage = {begin}; stage < {end}; ++stage) {{
        const int local = stage-{begin};
        const int stage_slot = stage == 0 ? 0 : (stage == 3 ? 2 : 1);
        const real c_dt = stage == 2 ? dt : half_dt;
        const real stage_scale = {scale_values};
{feature_setup}{term_time}        const int in_offset = (local & 1)*{planes}*area;
        const int out_offset = ((local+1) & 1)*{planes}*area;
        const real* bank = scratch+in_offset;
        #pragma unroll
        for (int it = 0; it < {iterations}; ++it) {{
            const int p = lane+it*{block};
            const int px = p%pitch, py = p/pitch;
            if (p >= area || px < local+1 || px >= pitch-local-1 ||
                {"py < local+1 || py >= rows-local-1" if two_d else "false"}) continue;
            const int cell = cells[it];
            real d[{planes}];
            real densities[2];
            real sums_x[2][2]{", sums_y[2][2], crs[2][2]" if two_d else ""};
            for (int c = 0; c < 2; ++c) {{
                const real re = bank[(2*c)*area+p], im = bank[(2*c+1)*area+p];
                densities[c] = re*re+im*im;
                for (int part = 0; part < 2; ++part) {{
                    const int plane = (2*c+part)*area;
                    const real centre = {cell("plane")};
                    sums_x[c][part] = {cell("plane", "+1")}+{cell("plane", "-1")}-2*centre;{"" if not two_d else f"""
                    sums_y[c][part] = {cell("plane", "+pitch")}+{cell("plane", "-pitch")}-2*centre;
                    crs[c][part] = {cell("plane", "+pitch+1")}-{cell("plane", "+pitch-1")}
                                  -{cell("plane", "-pitch+1")}+{cell("plane", "-pitch-1")};"""}
                }}
            }}
            for (int c = 0; c < 2; ++c) {{
                const int other = 1-c, rc = {comps} == 1 ? 0 : c;
                const int index = (member*2+c)*spatial_n+cell;
                const real re = bank[(2*c)*area+p], im = bank[(2*c+1)*area+p];
{operator_definitions}{cross_definition}                const real nr = {nr_load}, ni = {ni_load};
{relaxed_definition}{extra}                const real energy_h = ({potential_re}+gC*densities[c]+alpha2*densities[other]
                    +(c == 0 ? zeeman_half : -zeeman_half){gr_term}{gi_term}{extra_energy})*inv_hbar;
                const real gain = half_R*nr-half_gammaC+{potential_im}*inv_hbar{extra_rate};
                d[2*c] = gain*re+energy_h*im+{operator_re}{cross_re}{drive_re};
                d[2*c+1] = gain*im-energy_h*re+{operator_im}{cross_im}{drive_im};{reservoir_block}
            }}
            const int vx = base_x+px-{halo}, vy = base_y+py-{halo_y};
            const bool interior = px >= {halo} && px < {halo}+tx &&
                py >= {halo_y} && py < {halo_y}+{tile_y} && vx < nx && vy < ny;
            if (stage < 3) {{
                for (int k = 0; k < {planes}; ++k) {{
                    acc[it][k] = stage == 0 ? d[k] : acc[it][k]+(real)2*d[k];
                }}
                for (int k = 0; k < {planes}; ++k) {{
                    scratch[out_offset+k*area+p] = init[it][k]+c_dt*d[k];
                }}
{f"""                if (stage == {last} && interior) {{
{partial_write}                }}
""" if phase == "first" else ""}            }}{f"""
            else if (interior) {{
{final_write}            }}""" if phase != "first" else ""}
        }}
        __syncthreads();
    }}
}}
'''
    lines = source.split("\n")
    unrolled = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("for (int ") and "#pragma" not in unrolled[-1]:
            unrolled.append(line[:len(line) - len(stripped)] + "#pragma unroll")
        unrolled.append(line)
    return "\n".join(unrolled)


@register_solver("rk4-cuda")
class RK4CudaSolver(AbstractSolver):
    """Generic fused RK4 CUDA solver with overridable kernel and launch hooks."""
    capabilities = SolverCapabilities(
        components=frozenset({1, 2}), ndim=frozenset({1, 2}),
        grid_types=frozenset({"periodic", "closed-interval"}), reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        custom_reservoirs="unsupported", potential=True,
        kinetic_relaxation=True, dispersion=frozenset({"parabolic"}), terms=True,
        coherent_drive=True, time_dependent_potential=True,
        noise=frozenset({"none", "wigner"}),
        quantitative_reservoirs=frozenset({"single", "double", "quadratic-double", "none"}),
        description='GPU-fused CUDA RK4 solver. Stage-coupled reservoir. FDM production/reference path for periodic and closed-interval grids.',
    )

    def __init__(self, config: Config, grid: SimulationGrid, model):
        """Set up the rk 4 cuda solver."""
        super().__init__(config, grid, model)
        self._spinor = model.components == 2
        if self._spinor:
            self._init_spinor(config, grid, model)
            return
        self.ndim = grid.ndim
        self.nx = grid.nx
        self.ny = grid.ny if grid.ndim == 2 else None
        self.N = grid.nx * grid.ny if grid.ndim == 2 else grid.nx
        self.grid_type = grid.boundary
        _res_cfg = getattr(config, "reservoir", None)
        self._reservoir_type = (
            getattr(_res_cfg, "reservoir_type", "single") if _res_cfg else "single"
        )
        if self._reservoir_type not in ("single", "double", "quadratic-double", "none"):
            raise ValueError(
                f"rk4-cuda has hard-coded reservoir kernels and does not support "
                f"reservoir_type='{self._reservoir_type}'."
            )

        import numpy as _np

        self._model = model
        self._extended = (bool(model.terms) or model.time_dependent_potential
                          or model.drive is not None)
        self._noise = model.noise
        self._fused_noise = bool(model.noise is not None and model.noise.active)
        self._precision = getattr(config.solver, "precision", "double")
        if self._precision == "single":
            self._real_type = "float"
            self._np_real = _np.float32
            self._np_complex = _np.complex64
        else:
            self._real_type = "double"
            self._np_real = _np.float64
            self._np_complex = _np.complex128
        self._complex_dtype = self.xp.complex64 if self._precision == "single" else self.xp.complex128
        self._real_dtype = self.xp.float32 if self._precision == "single" else self.xp.float64

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
        self._inv_dy2 = 1.0 / (grid.dy ** 2) if grid.ndim == 2 else 0.0
        self._neg_hbar2_over_2m = -(self._hbar ** 2) / (2.0 * self._m_eff)
        self._inv_hbar = 1.0 / self._hbar

        laplacian = getattr(config.solver, "laplacian", "five-point")
        if laplacian not in ("five-point", "isotropic-9pt"):
            raise ValueError(
                f"solver.laplacian='{laplacian}' is not recognised. "
                "Valid values: 'five-point', 'isotropic-9pt'."
            )
        if laplacian == "isotropic-9pt" and grid.ndim == 1:
            raise ValueError("solver.laplacian='isotropic-9pt' is not supported in 1D")
        if laplacian == "isotropic-9pt":
            rel_diff = abs(grid.dx - grid.dy) / max(grid.dx, grid.dy)
            if rel_diff > 1e-8:
                raise ValueError(
                    f"solver.laplacian='isotropic-9pt' requires dx == dy, "
                    f"got dx={grid.dx}, dy={grid.dy} (relative difference {rel_diff:.2e})."
                )
        self._laplacian_type = laplacian

        self._use_gpu = hasattr(self.xp, "cuda")
        self._reference_features = self._extended and not self._use_gpu
        self._batched = False
        self._members = 1
        self._batch_ready = True
        self._physics_fields = tuple(name for name in self.batch_physics.host
                                     if name in ("hbar", "m_eff", "g_C", "g_R", "g_I", "R",
                                                 "gamma_C", "gamma_R", "gamma_I", "gamma_A",
                                                 "R_IA", "R_AI", "kappa", "kinetic_relaxation_eta",
                                                 "reservoir_diffusion_I", "reservoir_diffusion_A",
                                                 "reservoir_diffusion_R"))

        if self._use_gpu and not self._reference_features:
            self._physics_device = self.xp.zeros(1, dtype=self._np_real)
            _cast = self._np_real

            self._k_nx = _np.int32(self.nx)
            self._k_ny = _np.int32(self.ny if self.ny is not None else 0)
            self._k_N = _np.int32(self.N)
            self._k_int = _np.int32
            self._k_inv_dx2 = _cast(self._inv_dx2)
            self._k_inv_dy2 = _cast(self._inv_dy2)
            base = config.physics
            self._k_neg_hbar2_over_2m = _cast(-(base.hbar ** 2) / (2.0 * base.m_eff))
            self._k_inv_hbar = _cast(1.0 / base.hbar)
            self._k_g_C = _cast(base.g_C)
            self._k_g_R = _cast(base.g_R)
            self._k_g_I = _cast(base.g_I)
            self._k_R = _cast(base.R)
            self._k_gamma_C = _cast(base.gamma_C)
            self._k_gamma_R = _cast(base.gamma_R)

            self._k_half_dt = _cast(0.5 * self._dt)
            self._k_dt = _cast(self._dt)
            self._k_dt6 = _cast(self._dt / 6.0)
            self._k_kinetic_relaxation_eta = _cast(base.kinetic_relaxation_eta)
            self._k_diff_I = _cast(base.reservoir_diffusion_I)
            self._k_diff_A = _cast(base.reservoir_diffusion_A)
            self._k_diff_R = _cast(base.reservoir_diffusion_R)

            if self._reservoir_type == "double":
                self._k_gamma_I = _cast(base.gamma_I)
                self._k_gamma_A = _cast(base.gamma_A)
                self._k_R_IA = _cast(base.R_IA)
                self._k_R_AI = _cast(base.R_AI)
            elif self._reservoir_type == "quadratic-double":
                self._k_gamma_I = _cast(base.gamma_I)
                self._k_kappa = _cast(base.kappa)

            self._compile_kernels()
            self._init_gpu_buffers()

            common = (
                self._k_inv_dx2, self._k_inv_dy2,
                self._k_neg_hbar2_over_2m, self._k_inv_hbar,
                self._k_g_C, self._k_g_R,
            )
            if self._reservoir_type in ("single", "none"):
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
            self._use_tiled = (self._reservoir_type != "none" and not self._extended and (
                _prefer_tiled(self._precision, props["major"], props["minor"])
            ))
            self._tiled_kernels = {}
            self._tile_shape = (self._select_tile_shape() if self._use_tiled else None)
            self._graph_cache = {}
            self._graph_signature = None
            self._graphs_failed = False

    def _activate_batch(self, shape, batched):
        members = shape[0] if batched else 1
        if batched == self._batched and members == self._members and self._batch_ready:
            return
        import numpy as np

        self._batch_ready = False
        self._graph_cache.clear()
        self._graph_signature = None
        self._tiled_kernels.clear()
        for name in ("_graph_kernels", "_advance_buffers", "_graph_scales",
                     "_advance_stream", "_pump_combiner", "_zero_pump",
                     "_feature_prefix", "_feature_terms", "_feature_buffer"):
            if hasattr(self, name):
                delattr(self, name)
        self._k_psi = None
        self._k_reservoir = None
        self._potential_cached = None
        self._V_re_flat = None
        self._V_im_flat = None
        self._batched = batched
        self._members = members
        if batched:
            physics = self.config.physics
            values = lambda name: self.batch_physics.host.get(
                name, np.full(members, getattr(physics, name)))
            columns = {
                "neg_hbar2_over_2m": -(values("hbar") ** 2) / (2.0 * values("m_eff")),
                "inv_hbar": 1.0 / values("hbar"),
            }
            for field, name in (("g_C", "g_C"), ("g_R", "g_R"),
                                ("g_I", "g_I"), ("R", "R_cond"),
                                ("gamma_C", "gamma_C"), ("gamma_R", "gamma_R"),
                                ("gamma_I", "gamma_I"), ("gamma_A", "gamma_A"),
                                ("R_IA", "R_IA"), ("R_AI", "R_AI"),
                                ("kappa", "kappa"),
                                ("kinetic_relaxation_eta", "kinetic_relaxation_eta"),
                                ("reservoir_diffusion_R", "diff_R"),
                                ("reservoir_diffusion_A", "diff_A"),
                                ("reservoir_diffusion_I", "diff_I")):
                columns[name] = values(field)
            names = _batch_parameter_names(self._physics_fields)
            if names:
                self._physics_device = None
                self._physics_device = self.xp.asarray(
                    np.stack([columns[name] for name in names]),
                    dtype=self._np_real).ravel()
        self._compile_kernels()
        self._init_gpu_buffers()
        self._batch_ready = True

    def _init_spinor(self, config, grid, model):
        import numpy as np

        self._spinor_grid = grid
        self._spinor_dt = config.solver.dt
        self._spinor_real = np.float32 if config.solver.precision == "single" else np.float64
        self._spinor_complex = self.xp.complex64 if config.solver.precision == "single" else self.xp.complex128
        self._spinor_kernel = None
        self._spinor_tile_shape = None
        self._spinor_term_codegen = False
        self._spinor_term_arguments = ()
        self._spinor_graphs = {}
        self._spinor_graphs_failed = False
        if not hasattr(self.xp, "cuda"):
            return
        members = self.batch_physics.size or 1
        names = ("hbar", "m_eff", "g_C", "alpha_2", "zeeman", "te_tm",
                 "g_R", "g_I", "R", "gamma_C", "kinetic_relaxation_eta")
        columns = [np.asarray(self.batch_physics.host.get(
            name, np.full(members, getattr(config.physics, name))), dtype=self._spinor_real)
            for name in names]
        self._spinor_constants = self.xp.asarray(np.stack(columns, axis=1).ravel())
        reservoir_names = ("gamma_R", "gamma_I", "gamma_A", "R_IA", "R_AI", "kappa")
        reservoir_columns = [np.asarray(self.batch_physics.host.get(
            name, np.full(members, getattr(config.physics, name))), dtype=self._spinor_real)
            for name in reservoir_names]
        self._spinor_stage_constants = self.xp.asarray(
            np.stack(columns + reservoir_columns, axis=1).ravel())
        self._spinor_weights = (self.xp.ascontiguousarray(
            model.pump_weights, dtype=self._spinor_real).ravel()
            if model.pump_weights is not None else
            self.xp.ones(2 * members, dtype=self._spinor_real))
        self._spinor_feature_arrays = ()
        if model.time_dependent_potential or model.drive is not None:
            shape = (members, 2, *grid.shape)
            arrays = []
            if model.time_dependent_potential:
                cap = self.xp.broadcast_to(model.cap, shape)
                arrays.extend((self.xp.ascontiguousarray(
                    self.xp.real(cap), dtype=self._spinor_real).ravel(),
                    self.xp.ascontiguousarray(
                    self.xp.imag(cap), dtype=self._spinor_real).ravel()))
            if model.drive is not None:
                drive = self.xp.broadcast_to(self.xp.asarray(model.drive.spatial), shape)
                energy = self.xp.broadcast_to(self.xp.asarray(model.drive.energy), shape)
                arrays.extend((self.xp.ascontiguousarray(
                    self.xp.real(drive), dtype=self._spinor_real).ravel(),
                    self.xp.ascontiguousarray(
                    self.xp.imag(drive), dtype=self._spinor_real).ravel(),
                    self.xp.ascontiguousarray(energy, dtype=self._spinor_real).ravel()))
            self._spinor_feature_arrays = tuple(arrays)
            self._spinor_feature_values = self.xp.empty(64 * 3 * members * 3,
                                                         dtype=self._spinor_real)
        props = self.xp.cuda.runtime.getDeviceProperties(self.xp.cuda.runtime.getDevice())
        block_limit = props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"])
        sm_limit = props.get("sharedMemPerMultiprocessor", block_limit)
        real_type = "float" if config.solver.precision == "single" else "double"
        if grid.ndim == 1:
            tile_x = _choose_tile_x(real_type, "single", block_limit, sm_limit)
            self._spinor_tile_shape = (tile_x or 128, 1)
        else:
            self._spinor_tile_shape = (_choose_tile_shape(real_type, "single", block_limit,
                                                          sm_limit) or (8, 8))
        tile_x, tile_y = self._spinor_tile_shape
        self._spinor_term_codegen = bool(model.terms) and all(
            term.cuda is not None for term in model.terms)
        if self._spinor_term_codegen:
            xp = self.xp
            self._spinor_x_coords = xp.ascontiguousarray(grid.X, dtype=self._spinor_real).ravel()
            self._spinor_y_coords = (xp.ascontiguousarray(grid.Y, dtype=self._spinor_real).ravel()
                                     if grid.ndim == 2 else xp.zeros(grid.nx, dtype=self._spinor_real))
            arguments = []
            for term in model.terms:
                values = term.cuda_values()
                for ctype, name, *layout in term.cuda.arguments:
                    value = values[name]
                    if "*" in ctype:
                        dtype = (xp.uint64 if "long long" in ctype else
                                 xp.uint32 if "unsigned int" in ctype else
                                 xp.int32 if "int" in ctype else
                                 xp.float64 if "double" in ctype else
                                 xp.float32 if "float" in ctype else
                                 self._spinor_real)
                        arguments.append(xp.ascontiguousarray(value, dtype=dtype).ravel())
                    elif "int" in ctype:
                        arguments.append(np.int32(value))
                    else:
                        arguments.append(np.float64(value) if "double" in ctype else
                                         np.float32(value) if "float" in ctype else
                                         self._spinor_real(value))
            self._spinor_term_arguments = tuple(arguments)
        source = build_spinor_tiled_rhs_source(
            real_type, grid.boundary, tile_x, tile_y,
            model.terms if self._spinor_term_codegen else ())
        self._spinor_kernel = self.xp.RawKernel(source, "spinor_tiled_rhs")
        stage_source = build_spinor_tiled_stage_source(
            real_type, grid.boundary, tile_x, tile_y,
            model.terms if self._spinor_term_codegen else (),
            varying_potential=model.time_dependent_potential,
            drive=model.drive is not None)
        self._spinor_stage_kernel = self.xp.RawKernel(stage_source, "spinor_tiled_stage")
        self._spinor_graph_stage_kernel = self.xp.RawKernel(
            build_spinor_tiled_stage_source(real_type, grid.boundary, tile_x, tile_y,
                model.terms if self._spinor_term_codegen else (), graph=True,
                varying_potential=model.time_dependent_potential,
                drive=model.drive is not None),
            "spinor_tiled_stage")
        self._spinor_finish_kernel = self.xp.RawKernel(
            build_spinor_finish_source(real_type), "spinor_finish")
        relaxation = bool(self.batch_physics.any_nonzero("kinetic_relaxation_eta"))
        self._spinor_step_phases = ()
        self._spinor_step_kernels = {}
        reservoir_model = _SPINOR_RESERVOIR_MODELS.get(type(model.reservoir).__name__)
        if grid.boundary == "periodic" and reservoir_model is not None:
            planes = spinor_step_planes(reservoir_model, model.reservoir_components)
            phases = (("whole",) if _prefer_tiled(config.solver.precision, props["major"],
                                                  props["minor"]) else ("first", "second"))
            tiles = {phase: choose_spinor_step_tile(
                real_type, grid.ndim == 2, planes, block_limit, sm_limit,
                _SPINOR_STEP_STAGES[phase][1] - _SPINOR_STEP_STAGES[phase][0])
                for phase in phases}
            if all(tile is not None for tile in tiles.values()):
                self._spinor_step_phases = phases
                for phase, (tile_x_step, tile_y_step) in tiles.items():
                    halo = _SPINOR_STEP_STAGES[phase][1] - _SPINOR_STEP_STAGES[phase][0]
                    shared = _spinor_step_bytes(tile_x_step, tile_y_step, grid.ndim == 2,
                                                planes, real_type, halo)
                    for graph in (False, True):
                        kernel = self.xp.RawKernel(build_spinor_step_source(
                            real_type, tile_x_step, tile_y_step, grid.ndim == 2,
                            reservoir_model, model.reservoir_components,
                            model.terms if self._spinor_term_codegen else (), graph,
                            model.time_dependent_potential, model.drive is not None, phase,
                            relaxation),
                            "spinor_tiled_step")
                        if shared > props["sharedMemPerBlock"]:
                            kernel.max_dynamic_shared_size_bytes = shared
                        self._spinor_step_kernels[(phase, graph)] = (
                            kernel, (tile_x_step, tile_y_step), shared)
        self._spinor_buffers = None
        self._spinor_static_potential = None
        self._spinor_pump_cache = None
        if model.noise is not None and model.noise.active:
            from polarism.noise.philox import cuda_source

            noise_source = cuda_source(config.solver.precision)
            noise_source = noise_source.replace(
                "int member = j / spatial_n;",
                "int member = j / (2*spatial_n); int component = (j / spatial_n) % 2;")
            noise_source = noise_source.replace(
                "unsigned int d = 1U;", "unsigned int d = 1U + (unsigned int)component;")
            noise_source = noise_source.replace("0x1p-24f", "5.960464477539063e-8f")
            noise_source = noise_source.replace("0x1p-53", "1.1102230246251565e-16")
            noise_source = noise_source.replace("active[j]",
                f"active[(member*{model.reservoir_components}+(component*({model.reservoir_components}==2)))*spatial_n+j%spatial_n]")
            noise_source = noise_source.replace(
                "unsigned long long step, int spatial_n, int total_n,",
                "const unsigned long long* step_base, unsigned int step_offset, int spatial_n, int total_n,")
            noise_source = noise_source.replace(
                "    unsigned int b = (unsigned int)step;",
                "    unsigned long long step = step_base[0] + step_offset;\n"
                "    unsigned int b = (unsigned int)step;")
            self._spinor_noise_graph_kernel = self.xp.RawKernel(
                noise_source, f"add_wigner_{config.solver.precision}")
            self._spinor_noise_base = self.xp.zeros(1, dtype=self.xp.uint64)
        self._spinor_shared_bytes = 4 * (tile_x + 2) * (tile_y + 2) * self._spinor_real().nbytes

    def _spinor_rhs(self, model, field, reservoir_fields, t):
        from polarism.laser.pump_field import evaluate_pump
        import numpy as np

        if self._spinor_kernel is None:
            return model.rhs(field, reservoir_fields, t)
        xp = self.xp
        pump = evaluate_pump(model.pump, t)
        if model.reservoir_components == 1:
            density = (field.real * field.real + field.imag * field.imag).sum(axis=1, keepdims=True)
            res_derivs = model.reservoir.get_derivatives(field[:, :1], pump,
                                                          reservoir_fields, rho=density)
        else:
            res_derivs = model.reservoir.get_derivatives(field, pump * model.pump_weights,
                                                          reservoir_fields)
        active = model.reservoir.get_active_density(reservoir_fields)
        inactive = model.reservoir.get_inactive_density(reservoir_fields)
        zero = xp.zeros(field.shape, dtype=field.real.dtype)
        active = zero if xp.ndim(active) == 0 else xp.ascontiguousarray(xp.broadcast_to(active, field.shape))
        inactive = zero if xp.ndim(inactive) == 0 else xp.ascontiguousarray(xp.broadcast_to(inactive, field.shape))
        potential = model.potential_at(t)
        vr = xp.ascontiguousarray(xp.broadcast_to(xp.real(potential), field.shape))
        vi = (xp.ascontiguousarray(xp.broadcast_to(xp.imag(potential), field.shape))
              if xp.iscomplexobj(potential) else zero)
        result = xp.empty_like(field)
        tile_x, tile_y = self._spinor_tile_shape
        grid = self._spinor_grid
        arguments = (xp.ascontiguousarray(field).view(field.real.dtype).ravel(),
                     active.ravel(), inactive.ravel(), vr.ravel(), vi.ravel(),
                     self._spinor_constants, result.view(field.real.dtype).ravel(),
                     np.int32(grid.nx), np.int32(grid.ny if grid.ndim == 2 else 1),
                     self._spinor_real(1 / grid.dx**2),
                     self._spinor_real(1 / grid.dy**2 if grid.ndim == 2 else 0),
                     self._spinor_real(1 / (4 * grid.dx * grid.dy) if grid.ndim == 2 else 0))
        if self._spinor_term_codegen:
            arguments += (self._spinor_x_coords, self._spinor_y_coords,
                          self._spinor_real(t), *self._spinor_term_arguments)
        self._spinor_kernel(((grid.nx + tile_x - 1) // tile_x,
                             ((grid.ny + tile_y - 1) // tile_y if grid.ndim == 2 else 1),
                             field.shape[0]), (max(256, tile_x * tile_y),),
                            arguments,
                            shared_mem=self._spinor_shared_bytes)
        if model.terms and not self._spinor_term_codegen:
            result += model.extra_rhs(field, reservoir_fields, t)
        if model.drive is not None:
            result += model.drive_contribution(field, t)
        return result, res_derivs

    def _step_spinor(self, model, state, dt):
        self.batch_physics.check(state.field, 2)
        if self._spinor_fast_supported(model, state):
            return self._spinor_fast_step(model, state, dt)
        reservoir = model.reservoir
        field = state.field
        initial = reservoir.get_state()
        time = state.t
        half = 0.5 * dt
        k1, r1 = self._spinor_rhs(model, field, initial, time)
        k2, r2 = self._spinor_rhs(model, field + half * k1,
                                  tuple(a + half * b for a, b in zip(initial, r1)), time + half)
        k3, r3 = self._spinor_rhs(model, field + half * k2,
                                  tuple(a + half * b for a, b in zip(initial, r2)), time + half)
        k4, r4 = self._spinor_rhs(model, field + dt * k3,
                                  tuple(a + dt * b for a, b in zip(initial, r3)), time + dt)
        sixth = dt / 6.0
        result = field + sixth * (k1 + 2 * k2 + 2 * k3 + k4)
        state.field = model.boundary.after_step_action(result.astype(field.dtype, copy=False))
        reservoir.set_state(tuple(a + sixth * (b + 2 * c + 2 * d + e)
                                  for a, b, c, d, e in zip(initial, r1, r2, r3, r4)))
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED

    def _spinor_fast_supported(self, model, state):
        import numpy as np

        if self._spinor_kernel is None or (model.terms and not self._spinor_term_codegen):
            return False
        reservoir = model.reservoir
        if type(reservoir).__name__ not in _SPINOR_RESERVOIR_MODELS:
            return False
        if (hasattr(reservoir, "batch_physics") and
                any(reservoir.batch_physics.any_nonzero(name) for name in
                    ("reservoir_diffusion_R", "reservoir_diffusion_A", "reservoir_diffusion_I"))):
            return False
        terms = model.pump.separable_terms() if hasattr(model.pump, "separable_terms") else None
        if terms is None or len(terms) > 1:
            return False
        if terms and np.ndim(terms[0][1](state.t)) != 0:
            return False
        return True

    def _spinor_feature_samples(self, model, times):
        import numpy as np

        members = self.batch_physics.size or 1
        result = np.empty((len(times), members, 3), dtype=self._spinor_real)
        def sample(value):
            host = np.asarray(value.get() if hasattr(value, "get") else value)
            return np.broadcast_to(host.reshape(-1), (members,))
        for index, time in enumerate(times):
            result[index, :, 0] = (sample(model.potential_envelope(time))
                                   if model.time_dependent_potential else 1)
            result[index, :, 1] = (sample(model.drive.envelope(time))
                                   if model.drive is not None else 0)
            result[index, :, 2] = time
        return result.ravel()

    def _spinor_fast_step(self, model, state, dt, graph_scales=None,
                          graph_times=None, graph_offset=0):
        import numpy as np

        xp = self.xp
        field = state.field
        reservoir = model.reservoir
        initial = reservoir.get_state()
        res_components = model.reservoir_components
        grid = self._spinor_grid
        spatial_n = grid.nx * (grid.ny if grid.ndim == 2 else 1)
        shape_key = (field.shape, len(initial), res_components)
        if self._spinor_buffers is None or self._spinor_buffers[0] != shape_key:
            reservoir_shape = (field.shape[0], res_components, *grid.shape)
            zero_field = xp.zeros_like(field)
            zero_res = xp.zeros(reservoir_shape, dtype=self._spinor_real)
            staged = not self._spinor_step_phases
            field_derivatives = tuple(xp.empty_like(field) for _ in range(4 if staged else 0))
            res_derivatives = tuple(tuple(xp.empty(reservoir_shape, dtype=self._spinor_real)
                                          for _ in initial) for _ in range(4 if staged else 0))
            field_out = tuple(xp.empty_like(field) for _ in range(2))
            res_out = tuple(tuple(xp.empty_like(item) for item in initial)
                            for _ in range(2))
            partials = (tuple((xp.empty_like(field), tuple(xp.empty_like(item) for item in initial))
                              for _ in range(2)) if self._spinor_step_phases == ("first", "second")
                        else ())
            self._spinor_buffers = (shape_key, zero_field, zero_res,
                                    field_derivatives, res_derivatives, field_out, res_out,
                                    partials)
            self._spinor_graphs.clear()
        _, zero_field, zero_res, k, dr, outputs, reservoir_outputs, partials = self._spinor_buffers
        output_index = int(field.data.ptr == outputs[0].data.ptr)
        out = outputs[output_index]
        out_res = reservoir_outputs[output_index]
        potential_source = (model.potential_static if model.time_dependent_potential
                            else model.potential)
        potential_key = (id(potential_source), field.shape)
        if self._spinor_static_potential is None or self._spinor_static_potential[0] != potential_key:
            potential = xp.broadcast_to(potential_source, field.shape)
            vr = xp.ascontiguousarray(xp.real(potential), dtype=self._spinor_real).ravel()
            vi = xp.ascontiguousarray(xp.imag(potential), dtype=self._spinor_real).ravel()
            self._spinor_static_potential = (potential_key, vr, vi)
        _, vr, vi = self._spinor_static_potential
        terms = model.pump.separable_terms()
        if terms:
            envelope, coefficient = terms[0]
            if self._spinor_pump_cache is None or self._spinor_pump_cache[0] is not envelope:
                self._spinor_pump_cache = (
                    envelope, xp.ascontiguousarray(envelope, dtype=self._spinor_real).ravel())
            pump = self._spinor_pump_cache[1]
            scales = tuple(self._spinor_real(coefficient(t)) for t in
                           (state.t, state.t + dt / 2, state.t + dt))
        else:
            if not hasattr(self, "_spinor_zero_pump"):
                self._spinor_zero_pump = xp.zeros(spatial_n, dtype=self._spinor_real)
            pump = self._spinor_zero_pump
            scales = (self._spinor_real(1),) * 3
        model_id = _SPINOR_RESERVOIR_MODELS[type(reservoir).__name__]
        real = self._spinor_real
        has_features = model.time_dependent_potential or model.drive is not None
        if has_features and graph_scales is None:
            samples = self._spinor_feature_samples(
                model, (state.t, state.t + dt / 2, state.t + dt))
            self._spinor_feature_values[:samples.size].set(samples)
        field_view = field.view(field.real.dtype).ravel()
        a0 = initial[0].ravel() if initial else zero_res.ravel()
        b0 = initial[1].ravel() if len(initial) > 1 else zero_res.ravel()
        if self._spinor_step_phases:
            self._spinor_fused_step(model, state, dt, field, field_view, a0, b0, out,
                                    out_res, partials, vr, vi, pump, scales, graph_scales,
                                    graph_times, graph_offset, has_features, zero_res)
        else:
            self._spinor_staged_step(model, state, dt, field, field_view, a0, b0, k, dr,
                                     out, out_res, vr, vi, pump, scales, graph_scales,
                                     graph_times, graph_offset, spatial_n, model_id,
                                     has_features, zero_field, zero_res)
        field_n = field.size
        if graph_scales is not None and model.noise is not None and model.noise.active:
            noise = model.noise
            self._spinor_noise_graph_kernel(((field_n + 255) // 256,), (256,),
                (out.view(field.real.dtype).ravel(),
                 out_res[0].ravel() if initial else zero_res.ravel(),
                 noise._device_seeds, noise._device_gamma, noise._device_scattering,
                 noise._device_strength, self._spinor_noise_base,
                 np.uint32(graph_offset), np.int32(spatial_n), np.int32(field_n),
                 real(noise.dt / (4.0 * noise.volume))))
        state.field = model.boundary.after_step_action(out)
        for name, value in zip(reservoir.state_names, out_res):
            setattr(reservoir, name, value)
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED

    def _spinor_fused_step(self, model, state, dt, field, field_view, a0, b0, out,
                           out_res, partials, vr, vi, pump, scales, graph_scales,
                           graph_times, graph_offset, has_features, zero_res):
        import numpy as np

        grid = self._spinor_grid
        real = self._spinor_real
        two_d = grid.ndim == 2
        initial = model.reservoir.get_state()
        graph = graph_scales is not None
        reservoir_arrays = lambda items: (
            items[0].ravel() if initial else zero_res.ravel(),
            items[1].ravel() if len(initial) > 1 else zero_res.ravel())
        outputs = (out.view(field.real.dtype).ravel(), *reservoir_arrays(out_res))
        partial_arrays = tuple(
            (partial.view(field.real.dtype).ravel(), *reservoir_arrays(items))
            for partial, items in partials)
        common = (field_view, a0, b0, vr, vi, self._spinor_stage_constants, pump,
                  np.int32(pump.size), self._spinor_weights)
        geometry = (np.int32(grid.nx), np.int32(grid.ny if two_d else 1),
                    real(1 / grid.dx**2), real(1 / grid.dy**2 if two_d else 0),
                    real(1 / (4 * grid.dx * grid.dy) if two_d else 0),
                    real(dt / 2), real(dt), real(dt / 6))
        tail = scales if not graph else (graph_scales, np.int32(graph_offset * 3))
        if has_features:
            tail += (*self._spinor_feature_arrays, self._spinor_feature_values,
                     np.int32(graph_offset * 3 if graph else 0), np.int32(field.shape[0]))
        if self._spinor_term_codegen:
            tail += (self._spinor_x_coords, self._spinor_y_coords)
            if graph_times is None:
                tail += tuple(real(state.t + fraction * dt) for fraction in (0, 0.5, 1))
            else:
                tail += (graph_times, np.int32(graph_offset * 3))
            tail += self._spinor_term_arguments
        for phase in self._spinor_step_phases:
            kernel, (tile_x, tile_y), shared = self._spinor_step_kernels[(phase, graph)]
            produced = (outputs if phase == "whole" else
                        sum(partial_arrays, outputs if phase == "second" else ()))
            kernel(((grid.nx + tile_x - 1) // tile_x,
                    (grid.ny + tile_y - 1) // tile_y if two_d else 1, field.shape[0]),
                   (256,), common + produced + geometry + tail, shared_mem=shared)

    def _spinor_staged_step(self, model, state, dt, field, field_view, a0, b0, k, dr,
                            out, out_res, vr, vi, pump, scales, graph_scales,
                            graph_times, graph_offset, spatial_n, model_id, has_features,
                            zero_field, zero_res):
        import numpy as np

        real = self._spinor_real
        grid = self._spinor_grid
        initial = model.reservoir.get_state()
        res_components = model.reservoir_components
        tile_x, tile_y = self._spinor_tile_shape
        launch = ((grid.nx + tile_x - 1) // tile_x,
                  ((grid.ny + tile_y - 1) // tile_y if grid.ndim == 2 else 1),
                  field.shape[0])
        zero_view = zero_field.view(field.real.dtype).ravel()
        for stage in range(4):
            prior = stage - 1
            prev_field = k[prior].view(field.real.dtype).ravel() if stage else zero_view
            prev_active = dr[prior][0].ravel() if stage and initial else zero_res.ravel()
            prev_inactive = (dr[prior][1].ravel() if stage and len(initial) > 1
                             else zero_res.ravel())
            active_out = dr[stage][0].ravel() if initial else zero_res.ravel()
            inactive_out = dr[stage][1].ravel() if len(initial) > 1 else zero_res.ravel()
            factor = real((0, dt / 2, dt / 2, dt)[stage])
            args = (field_view, prev_field, a0, b0, prev_active, prev_inactive,
                    vr, vi, self._spinor_stage_constants,
                    k[stage].view(field.real.dtype).ravel(), np.int32(grid.nx),
                    np.int32(grid.ny if grid.ndim == 2 else 1),
                    real(1 / grid.dx**2),
                    real(1 / grid.dy**2 if grid.ndim == 2 else 0),
                    real(1 / (4 * grid.dx * grid.dy) if grid.ndim == 2 else 0),
                    pump, np.int32(pump.size), self._spinor_weights,
                    np.int32(res_components), np.int32(model_id), factor)
            if graph_scales is None:
                args += (scales[(0, 1, 1, 2)[stage]],)
            else:
                args += (graph_scales, np.int32(graph_offset * 3 +
                                               (0, 1, 1, 2)[stage]))
            args += (active_out, inactive_out)
            if has_features:
                args += (*self._spinor_feature_arrays, self._spinor_feature_values,
                         np.int32((graph_offset * 3 if graph_scales is not None else 0) +
                                  (0, 1, 1, 2)[stage]), np.int32(field.shape[0]))
            if self._spinor_term_codegen:
                args += (self._spinor_x_coords, self._spinor_y_coords)
                if graph_times is None:
                    args += (real(state.t + (0, 0.5, 0.5, 1)[stage] * dt),)
                else:
                    args += (graph_times, np.int32(graph_offset * 3 +
                                                  (0, 1, 1, 2)[stage]))
                args += self._spinor_term_arguments
            kernel = (self._spinor_graph_stage_kernel if graph_scales is not None
                      else self._spinor_stage_kernel)
            kernel(launch, (max(256, tile_x * tile_y),), args,
                   shared_mem=self._spinor_shared_bytes)
        field_n = field.size
        reservoir_n = field.shape[0] * res_components * spatial_n if initial else 0
        a = tuple(item[0].ravel() if initial else zero_res.ravel() for item in dr)
        b = tuple(item[1].ravel() if len(initial) > 1 else zero_res.ravel()
                  for item in dr)
        self._spinor_finish_kernel(((2 * field_n + 255) // 256,), (256,),
            (field_view, *(item.view(field.real.dtype).ravel() for item in k),
             a0, *a, b0, *b, out.view(field.real.dtype).ravel(),
             out_res[0].ravel() if initial else zero_res.ravel(),
             out_res[1].ravel() if len(initial) > 1 else zero_res.ravel(),
             np.int32(field_n), np.int32(reservoir_n), np.int32(model_id), real(dt / 6)))

    def _advance_spinor_graph(self, model, state, n_steps):
        import numpy as np

        if n_steps <= 0:
            return
        dt = self.config.solver.dt
        self._spinor_fast_step(model, state, dt)
        if model.noise is not None and model.noise.active:
            model.noise.apply_to_state(state, model.reservoir, state.step - 1)
        if n_steps == 1:
            return
        xp = self.xp
        if not hasattr(self, "_spinor_graph_stream"):
            self._spinor_graph_stream = xp.cuda.Stream(non_blocking=True)
            self._spinor_graph_scales = xp.empty(64 * 3, self._spinor_real)
            self._spinor_graph_times = xp.empty(64 * 3, self._spinor_real)
        self._spinor_graph_stage_kernel.compile()
        self._spinor_finish_kernel.compile()
        if model.noise is not None and model.noise.active:
            self._spinor_noise_graph_kernel.compile()
        xp.cuda.get_current_stream().synchronize()
        stream = self._spinor_graph_stream
        term = model.pump.separable_terms()
        coefficient = term[0][1] if term else None
        remaining = n_steps - 1
        pending_hosts = []
        with stream:
            while remaining:
                length = min(64, remaining)
                reservoir = model.reservoir
                initial = reservoir.get_state()
                key = (length, state.field.data.ptr,
                       tuple(item.data.ptr for item in initial))
                if key not in self._spinor_graphs:
                    start_step, start_time = state.step, state.t
                    start_field = state.field
                    stream.synchronize()
                    stream.begin_capture()
                    try:
                        for offset in range(length):
                            self._spinor_fast_step(model, state, dt,
                                self._spinor_graph_scales, self._spinor_graph_times,
                                offset)
                        graph = stream.end_capture()
                        final_field = state.field
                        final_reservoir = reservoir.get_state()
                    except Exception:
                        try:
                            stream.end_capture()
                        except Exception:
                            pass
                        state.step, state.t = start_step, start_time
                        state.field = start_field
                        for name, value in zip(reservoir.state_names, initial):
                            setattr(reservoir, name, value)
                        raise
                    state.step, state.t = start_step, start_time
                    state.field = start_field
                    for name, value in zip(reservoir.state_names, initial):
                        setattr(reservoir, name, value)
                    self._spinor_graphs[key] = (graph, final_field, final_reservoir)
                times = np.asarray([state.t0+(state.step+k)*dt+c*dt
                                    for k in range(length) for c in (0, 0.5, 1)])
                scales = (np.asarray([coefficient(t) for t in times],
                                     dtype=self._spinor_real) if coefficient is not None
                          else np.ones(length * 3, dtype=self._spinor_real))
                self._spinor_graph_scales[:length*3].set(scales, stream=stream)
                pending_hosts.append(scales)
                if self._spinor_term_codegen:
                    stage_times = times.astype(self._spinor_real)
                    self._spinor_graph_times[:length*3].set(stage_times, stream=stream)
                    pending_hosts.append(stage_times)
                if model.time_dependent_potential or model.drive is not None:
                    feature_values = self._spinor_feature_samples(model, times)
                    self._spinor_feature_values[:feature_values.size].set(
                        feature_values, stream=stream)
                    pending_hosts.append(feature_values)
                if model.noise is not None and model.noise.active:
                    step_base = np.asarray([state.step], dtype=np.uint64)
                    self._spinor_noise_base.set(step_base, stream=stream)
                    pending_hosts.append(step_base)
                graph, final_field, final_reservoir = self._spinor_graphs[key]
                graph.launch(stream=stream)
                state.field = final_field
                for name, value in zip(reservoir.state_names, final_reservoir):
                    setattr(reservoir, name, value)
                state.step += length
                state.t = state.t0 + state.step * dt
                remaining -= length
                if len(pending_hosts) >= 32:
                    stream.synchronize()
                    pending_hosts.clear()
            stream.synchronize()

    def _build_kernel_source(
        self, bc_type: str, real_type: str, reservoir_type: str
    ) -> tuple[str, str, str]:
        """Build the CUDA kernel source."""
        builder = build_flat_1d_sources if self.ndim == 1 else build_kernel_source_1d
        sources = builder(
            bc_type, real_type,
            "single" if reservoir_type == "none" else reservoir_type,
            **({"laplacian_type": self._laplacian_type} if self.ndim == 2 else {}),
            batched=self._batched, physics_fields=self._physics_fields,
        )
        if reservoir_type == "none":
            sources = tuple(_without_reservoir(source, real_type) for source in sources)
        if self._extended:
            sources = tuple(_extend_cuda_source(
                source, real_type, reservoir_type, self._batched,
                self._model.terms, self._model.time_dependent_potential,
                self._model.drive is not None,
            ) for source in sources)
        return sources

    def _select_block_size(self) -> int:
        """Pick the CUDA block size."""
        return _auto_tune_block_size(self._kern_rhs, self.N * self._members)

    def _make_launch_dims(self, block_size: int) -> None:
        """Set the CUDA launch dimensions."""
        grid = (self.N + block_size - 1) // block_size
        self._k_grid = (grid, self._members) if self._batched else (grid,)
        self._k_block = (block_size,)

    def _compile_kernels(self) -> None:
        """Compile the CUDA kernels."""
        import cupy as cp

        rhs_src, fused_src, final_src = self._build_kernel_source(
            self.grid_type, self._real_type, self._reservoir_type
        )

        if self._reservoir_type == "none":
            self._kern_rhs = cp.RawKernel(rhs_src, "gpe_rhs_none")
            self._kern_fused = cp.RawKernel(fused_src, "fused_stage_rhs_none")
            self._kern_final = cp.RawKernel(final_src, "fused_final_stage_none")
        elif self._reservoir_type == "double":
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

        if self._fused_noise and not self._reference_features:
            suffix = {"single": "", "double": "_double",
                      "quadratic-double": "_qdouble", "none": "_none"}[self._reservoir_type]
            source = _fused_noise_source(final_src, self._real_type,
                                         self._reservoir_type, self._batched, False)
            self._kern_final_noise = cp.RawKernel(source, "fused_final_stage" + suffix)

        block = self._select_block_size()
        self._make_launch_dims(block)

    def _select_tile_shape(self):
        """Choose a tile that fits device shared memory, favoring two resident blocks."""
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        block_limit = props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"])
        sm_limit = props.get("sharedMemPerMultiprocessor", block_limit)
        if self.ndim == 1:
            tile_x = _choose_tile_x(self._real_type, self._reservoir_type,
                                    block_limit, sm_limit)
            return (tile_x, 1) if tile_x is not None else None
        return _choose_tile_shape(self._real_type, self._reservoir_type,
                                  block_limit, sm_limit)

    def _tiled_kernel(self, complex_potential, graph=False, noise=False):
        key = (complex_potential, graph, noise)
        if key not in self._tiled_kernels:
            import cupy as cp

            tile_x, tile_y = self._tile_shape
            if self.ndim == 1:
                source = build_tiled_1d_kernel_source(
                    self.grid_type, self._real_type, self._reservoir_type,
                    tile_x, complex_potential, graph, self._batched,
                    self._physics_fields,
                )
            else:
                source = build_tiled_kernel_source(
                    self.grid_type, self._real_type, self._reservoir_type,
                    self._laplacian_type, tile_x, tile_y, complex_potential, graph,
                    self._batched, self._physics_fields,
                )
            if noise:
                source = _tiled_wigner_source(source, self._real_type,
                                              self._reservoir_type, self._batched,
                                              graph)
            suffix = {"single": "", "double": "_double",
                      "quadratic-double": "_qdouble"}[self._reservoir_type]
            name = "tiled_step_1d" if self.ndim == 1 else "tiled_step"
            kernel = cp.RawKernel(source, name + suffix)
            shared_bytes = (_tile_shared_bytes_1d(tile_x, self._real_type,
                                                   self._reservoir_type)
                            if self.ndim == 1 else
                            _tile_shared_bytes(tile_x, tile_y, self._real_type,
                                               self._reservoir_type))
            if shared_bytes > cp.cuda.runtime.getDeviceProperties(
                    cp.cuda.runtime.getDevice())["sharedMemPerBlock"]:
                kernel.max_dynamic_shared_size_bytes = shared_bytes
            kernel.compile()
            self._tiled_kernels[key] = (kernel, shared_bytes)
        return self._tiled_kernels[key]

    def _init_gpu_buffers(self) -> None:
        """Allocate three RK4 derivative stages and constant potential buffer."""
        xp = self.xp
        N = self.N * self._members

        self._complex_dtype = xp.complex64 if self._precision == "single" else xp.complex128
        self._real_dtype = xp.float32 if self._precision == "single" else xp.float64
        n_fields = 0 if self._reservoir_type == "none" else (1 if self._reservoir_type == "single" else 2)
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

    def _prepare_feature_buffers(self, shape):
        """Keep static CUDA snippet inputs and per-stage device scalars."""
        if not self._extended or hasattr(self, "_feature_prefix"):
            return
        import numpy as np

        xp = self.xp
        model = self._model
        prefix = []
        if model.time_dependent_potential:
            static = xp.asarray(model.potential_static)
            if static.ndim == 0:
                static = xp.broadcast_to(static, model.grid.shape)
            cap = xp.broadcast_to(xp.asarray(model.cap), static.shape)
            self._extended_v_re = xp.ascontiguousarray(xp.real(static), dtype=self._real_dtype).ravel()
            self._extended_v_im = xp.ascontiguousarray(xp.imag(static), dtype=self._real_dtype).ravel()
            prefix.extend((xp.ascontiguousarray(xp.real(cap), dtype=self._real_dtype).ravel(),
                           xp.ascontiguousarray(xp.imag(cap), dtype=self._real_dtype).ravel()))
        if model.drive is not None:
            spatial = xp.broadcast_to(xp.asarray(model.drive.spatial), shape)
            energy = xp.broadcast_to(xp.asarray(model.drive.energy), shape)
            prefix.extend((xp.ascontiguousarray(xp.real(spatial), dtype=self._real_dtype).ravel(),
                           xp.ascontiguousarray(xp.imag(spatial), dtype=self._real_dtype).ravel(),
                           xp.ascontiguousarray(energy, dtype=self._real_dtype).ravel()))
        if model.terms:
            prefix.append(xp.ascontiguousarray(model.grid.X, dtype=self._real_dtype).ravel())
            prefix.append(xp.ascontiguousarray(model.grid.Y, dtype=self._real_dtype).ravel()
                          if self.ndim == 2 else xp.zeros(self.N, dtype=self._real_dtype))
        term_arguments = []
        for term in model.terms:
            values = term.cuda_values()
            for argument in term.cuda.arguments:
                ctype, name = argument[:2]
                value = values[name]
                if "*" in ctype:
                    dtype = (xp.uint64 if "long long" in ctype else
                             xp.uint32 if "unsigned int" in ctype else
                             xp.int32 if "int" in ctype else
                             xp.float64 if "double" in ctype else
                             xp.float32 if "float" in ctype else self._real_dtype)
                    term_arguments.append(xp.ascontiguousarray(value, dtype=dtype).ravel())
                elif "int" in ctype:
                    term_arguments.append(np.int32(value))
                else:
                    term_arguments.append(np.float64(value) if "double" in ctype else np.float32(value)
                                          if "float" in ctype else self._np_real(value))
        self._feature_prefix = tuple(prefix)
        self._feature_terms = tuple(term_arguments)
        self._feature_buffer = xp.empty(64 * 3 * self._members * 3,
                                        dtype=self._real_dtype)

    def _feature_values(self, times, stream=None):
        """Copy potential, drive, and stage-time coefficients to the device."""
        import numpy as np

        model = self._model
        values = np.empty((len(times), self._members, 3), dtype=self._np_real)
        def members(value):
            host = np.asarray(value.get() if hasattr(value, "get") else value)
            if host.ndim == 0:
                return np.full(self._members, host.item(), dtype=self._np_real)
            return np.broadcast_to(host.reshape(-1), (self._members,))
        for stage, time in enumerate(times):
            values[stage, :, 0] = (members(model.potential_envelope(time))
                                   if model.time_dependent_potential else 1)
            values[stage, :, 1] = (members(model.drive.envelope(time))
                                   if model.drive is not None else 0)
            values[stage, :, 2] = time
        self._feature_buffer[:values.size].set(values.ravel(), stream=stream)
        return values

    def _step_gpu(self, psi0, reservoir0, V_re, V_im, pumps, pump_scales,
                  v_im_scale, psi_out, reservoir_out, scale_buffer=None,
                  scale_offset=0, noise_step=None, noise_graph_offset=0,
                  feature_offset=0):
        """Launch the same RK4 kernels for each supported reservoir model."""
        if self._batched:
            sizes = (self.N, self.N * self._members)
            if V_re.size not in sizes or V_im.size not in sizes or any(
                    pump.size not in sizes for pump in pumps):
                raise ValueError("batched rk4-cuda expects shared 2-D or member 3-D pump and potential")
        def batch_args(stage):
            return ((self._k_int(self._members), self._k_int(V_re.size),
                     self._k_int(pumps[stage].size), self._physics_device)
                    if self._batched else ())
        def feature_args(stage):
            return ((*self._feature_prefix, self._feature_buffer,
                     self._k_int(feature_offset + stage), self._k_int(self._members),
                     *self._feature_terms) if self._extended else ())
        apply_noise = self._fused_noise and (noise_step is not None or scale_buffer is not None)
        noise_args = ((self._noise._device_seeds, self._noise._device_gamma,
                       self._noise._device_scattering, self._noise._device_strength,
                       self._np_real(self._noise.dt / (4.0 * self._noise.volume)),
                       *((self._noise_step_base, self._k_int(noise_graph_offset))
                         if scale_buffer is not None else (self.xp.uint64(noise_step),)))
                      if apply_noise else ())
        if self._use_tiled and self._tile_shape is not None:
            kernel, shared_bytes = self._tiled_kernel(bool(v_im_scale),
                                                       scale_buffer is not None,
                                                       apply_noise)
            tile_x, tile_y = self._tile_shape
            scale_args = ((scale_buffer, self._k_int(scale_offset))
                          if scale_buffer is not None else pump_scales)
            tile_grid = (((self.nx + tile_x - 1) // tile_x, self._members)
                         if self.ndim == 1 else
                         ((self.nx + tile_x - 1) // tile_x,
                          (self.ny + tile_y - 1) // tile_y, self._members))
            kernel(
                tile_grid, (256,),
                (psi0, *reservoir0, V_re, V_im, *pumps,
                 psi_out, *reservoir_out, self._k_nx, self._k_ny,
                 *scale_args, v_im_scale, self._k_half_dt, self._k_dt,
                 self._k_dt6, *self._k_rhs_parameters,
                 *((self._k_int(self._members), self._k_int(V_re.size),
                    *(self._k_int(pump.size) for pump in pumps), self._physics_device)
                   if self._batched else ()), *noise_args),
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
            (self._kern_rhs, self._kern_fused,
             self._kern_final_noise if apply_noise else self._kern_final)
        )

        def scale_args(stage):
            if scale_buffer is None:
                return (pump_scales[stage],)
            return (scale_buffer, self._k_int(scale_offset + stage))

        rhs(
            grid, block,
            (psi0, *reservoir0, V_re, V_im, pumps[0],
             k_psi[0], *k_reservoir[0], *dimensions,
             *scale_args(0), v_im_scale, *parameters, *batch_args(0), *feature_args(0)),
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
                 self._k_half_dt, *parameters, *batch_args(1), *feature_args(1)),
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
             self._k_dt, self._k_dt6, *parameters, *batch_args(2), *feature_args(2),
             *noise_args),
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
                  "quadratic-double": "_qdouble", "none": "_none"}[self._reservoir_type]
        kernels = []
        for source, name in zip(sources, names):
            source = source.replace(
                f"const {self._real_type} pump_scale,",
                f"const {self._real_type}* pump_scales, const int pump_index,",
            )
            source = re.sub(r"\bpump_scale\b", "pump_scales[pump_index]", source)
            if name == "fused_final_stage" and self._fused_noise:
                source = _fused_noise_source(source, self._real_type,
                                             self._reservoir_type, self._batched, True)
            kernel = self.xp.RawKernel(source, name + suffix)
            kernel.compile()
            kernels.append(kernel)
        self._graph_kernels = tuple(kernels)

    def _ensure_advance_buffers(self):
        if hasattr(self, "_advance_buffers"):
            return
        xp = self.xp
        fields = 0 if self._reservoir_type == "none" else (1 if self._reservoir_type == "single" else 2)
        total = self.N * self._members
        psi = tuple(xp.empty(total, self._complex_dtype) for _ in range(2))
        res = tuple(tuple(xp.empty(total, self._real_dtype) for _ in range(fields))
                    for _ in range(2))
        graph_scales = xp.empty(64 * 3, self._real_dtype)
        advance_stream = xp.cuda.Stream(non_blocking=True)
        self._advance_buffers = (psi, res)
        self._graph_scales = graph_scales
        if getattr(self, "_fused_noise", False):
            self._noise_step_base = xp.zeros(1, dtype=xp.uint64)
        self._advance_stream = advance_stream

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
            self._tiled_kernel(bool(v_im_scale), graph=True,
                               noise=self._fused_noise)
        else:
            self._compile_graph_kernels()
        psi, res = self._advance_buffers
        scales = self._graph_scales
        vr, vi = ((self._extended_v_re, self._extended_v_im)
                  if self._extended and self._model.time_dependent_potential
                  else self._prepare_potential_buffers(potential))
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
                        noise_graph_offset=k,
                        feature_offset=3 * k,
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

    def advance(self, model, state, n_steps: int) -> None:
        """Run CUDA chunks; custom reservoirs or boundary actions need individual steps."""
        if self._spinor:
            if (self.config.solver.cuda_graphs and not self._spinor_graphs_failed and
                    getattr(model.boundary.absorption, "after_step_is_noop", False) and
                    self._spinor_fast_supported(model, state)):
                field_backup = state.field.copy()
                reservoir_backup = tuple(value.copy() for value in model.reservoir.get_state())
                start_step, start_time = state.step, state.t
                try:
                    return self._advance_spinor_graph(model, state, n_steps)
                except Exception as exc:
                    import sys

                    self._spinor_graphs_failed = True
                    self._spinor_graphs.clear()
                    print(f"rk4-cuda: spinor CUDA graphs unavailable ({exc}); using regular launches.",
                          file=sys.stderr)
                    self.xp.cuda.get_current_stream().synchronize()
                    state.field = field_backup
                    for name, value in zip(model.reservoir.state_names, reservoir_backup):
                        setattr(model.reservoir, name, value)
                    state.step, state.t = start_step, start_time
                    return super().advance(model, state, n_steps)
            return super().advance(model, state, n_steps)
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        start_step = state.step
        if n_steps <= 0:
            return
        self.batch_physics.check(state.field)
        if not self._use_gpu or self._reference_features:
            return super().advance(model, state, n_steps)
        self._activate_batch(state.field.shape, state.batched)
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.no_reservoir import NoReservoir

        supported = {"single": SingleReservoir, "double": DoubleReservoir,
                     "quadratic-double": QuadraticDoubleReservoir,
                     "none": NoReservoir}
        if (type(reservoir) is not supported[self._reservoir_type]
                or not getattr(boundary_condition.absorption, "after_step_is_noop", False)):
            return super().advance(model, state, n_steps)

        xp = self.xp
        shape = state.field.shape
        self._prepare_feature_buffers(shape)
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        term = (terms[0] if terms is not None and len(terms) == 1 else
                pump.separable_term() if terms is None and hasattr(pump, "separable_term") else None)
        if terms is None and term is not None:
            terms = (term,)
        multi = terms is not None and (len(terms) > 1 or
                    (state.batched and len(terms) == 1 and
                     self.xp.ndim(terms[0][1](state.t)) != 0))
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
        vr, vi = ((self._extended_v_re, self._extended_v_im)
                  if self._extended and model.time_dependent_potential
                  else self._prepare_potential_buffers(potential))
        vi_scale = self._np_real(1 if xp.iscomplexobj(potential) else 0)
        current_psi = self._flat_gpu_input(state.field, self._complex_dtype)
        current_res = tuple(self._flat_gpu_input(x, self._real_dtype)
                            for x in reservoir.get_state())
        xp.cuda.get_current_stream().synchronize()
        self._ensure_advance_buffers()
        if multi:
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if not hasattr(self, "_pump_combiner"):
                self._pump_combiner = CudaPumpCombiner(xp, self._real_dtype,
                                                       self.N * self._members)
            combiner = self._pump_combiner
            combiner.prepare(terms, 3, 2 * 64 + 1, shape=shape if state.batched else None)
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
                            state.t0 + (start_step + done + k) * self._dt,
                            state.t0 + (start_step + done + k) * self._dt + 0.5 * self._dt,
                            state.t0 + (start_step + done + k) * self._dt + self._dt)]
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
                        if self._extended:
                            pending_hosts.append(self._feature_values(times, stream))
                        if self._fused_noise:
                            host_step = __import__("numpy").asarray(
                                [start_step + done], dtype="uint64")
                            pending_hosts.append(host_step)
                            self._noise_step_base.set(host_step, stream=stream)
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
                t = state.t0 + (start_step + k) * self._dt
                times = (t, t + 0.5 * self._dt, t + self._dt)
                if self._extended:
                    pending_hosts.append(self._feature_values(times, stream))
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
                               work_psi[1 - parity], work_res[1 - parity],
                               noise_step=start_step + k)
                if len(pending_hosts) >= 32:
                    stream.synchronize()
                    pending_hosts.clear()
            final_index = (n_steps - done) % 2
            state.field = boundary_condition.after_step_action(
                work_psi[final_index].reshape(shape).copy()
            )
            reservoir.set_state(tuple(field.reshape(shape).copy()
                                      for field in work_res[final_index]))
            stream.synchronize()
        state.step = start_step + n_steps
        state.t = state.t0 + state.step * self._dt

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
        if self.ndim == 1:
            if self.grid_type == "periodic":
                out[:] = (self.xp.roll(psi, -1, axis=-1) - 2 * psi
                          + self.xp.roll(psi, 1, axis=-1)) * self._inv_dx2
            else:
                out[..., 1:-1] = (psi[..., 2:] - 2 * psi[..., 1:-1]
                                  + psi[..., :-2]) * self._inv_dx2
                out[..., 0] = 2 * (psi[..., 1] - psi[..., 0]) * self._inv_dx2
                out[..., -1] = 2 * (psi[..., -2] - psi[..., -1]) * self._inv_dx2
            return
        if self._laplacian_type == "isotropic-9pt":
            if psi.ndim == 2:
                self._cpu_laplacian_9pt(psi, out)
            else:
                for member in range(psi.shape[0]):
                    self._cpu_laplacian_9pt(psi[member, 0], out[member, 0])
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
        p0 = evaluate_pump(pump, t)
        p_half = evaluate_pump(pump, t + 0.5 * dt)
        p1 = evaluate_pump(pump, t + dt)
        psi0 = state.field.copy()
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
        state.field = boundary_condition.after_step_action(
            psi0 + dt6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        )
        reservoir.set_state(tuple(
            r0 + dt6 * (r1 + 2 * r2 + 2 * r3 + r4)
            for r0, r1, r2, r3, r4 in zip(res0, k1r, k2r, k3r, k4r)
        ))

    def step(self, model, state, dt):
        """Advance with CUDA kernels or CuPy stages for a custom reservoir model."""
        if self._spinor:
            return self._step_spinor(model, state, dt)
        potential = model.potential
        pump = model.pump
        reservoir = model.reservoir
        boundary_condition = model.boundary
        self.batch_physics.check(state.field)
        if self._reference_features:
            return self._step_reference(model, state, dt)
        if not self._use_gpu:
            self._step_cpu(potential, pump, reservoir, boundary_condition, state)
            state.step += 1
            state.t = state.t0 + state.step * dt
            return ACCEPTED
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.no_reservoir import NoReservoir

        supported = {"single": SingleReservoir, "double": DoubleReservoir,
                     "quadratic-double": QuadraticDoubleReservoir,
                     "none": NoReservoir}
        if type(reservoir) is not supported[self._reservoir_type]:
            self._step_cpu(potential, pump, reservoir, boundary_condition, state)
            state.step += 1
            state.t = state.t0 + state.step * dt
            return ACCEPTED
        self._activate_batch(state.field.shape, state.batched)

        shape = state.field.shape
        self._prepare_feature_buffers(shape)
        V_re_flat, V_im_flat = ((self._extended_v_re, self._extended_v_im)
                                 if self._extended and model.time_dependent_potential
                                 else self._prepare_potential_buffers(potential))
        t = state.t
        dt = self._dt
        stage_times = (t, t + 0.5 * dt, t + dt)
        if self._extended:
            self._feature_host = self._feature_values(stage_times,
                                                       self.xp.cuda.get_current_stream())
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        separable = (terms[0] if terms is not None and len(terms) == 1 else
                     pump.separable_term() if terms is None and hasattr(pump, "separable_term")
                     else None)
        if terms is None and separable is not None:
            terms = (separable,)
        combined = terms is not None and (len(terms) > 1 or
                   (len(terms) == 1 and state.batched and
                    self.xp.ndim(terms[0][1](t)) != 0))
        if combined:
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if not hasattr(self, "_pump_combiner"):
                self._pump_combiner = CudaPumpCombiner(self.xp, self._real_dtype,
                                                       self.N * self._members)
            combiner = self._pump_combiner
            combiner.prepare(terms, 3, 3, shape=shape if state.batched else None)
            stream = self.xp.cuda.get_current_stream()
            combiner.coefficients(terms, stage_times, stream)
            stream.synchronize()
            pumps = tuple(combiner.combine(i, i) for i in range(3))
            pump_scales = (self._np_real(1),) * 3
        elif separable is None:
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

        psi0 = self._flat_gpu_input(state.field, self._complex_dtype)
        reservoir_state = reservoir.get_state()
        reservoir0 = tuple(
            self._flat_gpu_input(field, self._real_dtype)
            for field in reservoir_state
        )
        psi_out = self.xp.empty(self.N * self._members, dtype=self._complex_dtype)
        reservoir_out = tuple(
            self.xp.empty(self.N * self._members, dtype=self._real_dtype)
            for _ in reservoir0
        )

        # All producers, RK4 kernels, and consumers use the caller's current stream.
        self._step_gpu(
            psi0, reservoir0, V_re_flat, V_im_flat, pumps, pump_scales,
            v_im_scale, psi_out, reservoir_out
        )
        state.field = boundary_condition.after_step_action(psi_out.reshape(shape))
        reservoir.set_state(tuple(field.reshape(shape) for field in reservoir_out))
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED

    def _step_reference(self, model, state, dt):
        """Evaluate extended model features at all four RK stages."""
        reservoir = model.reservoir
        field = state.field
        initial = reservoir.get_state()
        time = state.t
        half = 0.5 * dt
        k1, r1 = model.rhs(field, initial, time)
        k2, r2 = model.rhs(field + half * k1,
                            tuple(a + half * b for a, b in zip(initial, r1)),
                            time + half)
        k3, r3 = model.rhs(field + half * k2,
                            tuple(a + half * b for a, b in zip(initial, r2)),
                            time + half)
        k4, r4 = model.rhs(field + dt * k3,
                            tuple(a + dt * b for a, b in zip(initial, r3)),
                            time + dt)
        sixth = dt / 6.0
        result = field + sixth * (k1 + 2 * k2 + 2 * k3 + k4)
        state.field = model.boundary.after_step_action(result.astype(field.dtype, copy=False))
        reservoir.set_state(tuple(
            a + sixth * (b + 2 * c + 2 * d + e)
            for a, b, c, d, e in zip(initial, r1, r2, r3, r4)
        ))
        state.step += 1
        state.t = state.t0 + state.step * dt
        return ACCEPTED
