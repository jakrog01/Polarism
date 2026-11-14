"""Fused CUDA stages and captured fixed-step chunks for Lawson RK."""
from __future__ import annotations

import numpy as np
import re

from polarism.laser.pump_field import evaluate_pump


_SOURCE = r"""
typedef {real} real;
#define MODEL {model}
#define ETA {eta}
#define STAGES {stages}
#define FIELDS {fields}
__device__ __forceinline__ void phase(real theta, real* sn, real* cs) {{
    {sincos}(theta, sn, cs);
}}
extern "C" __global__ void combine_k(
    const real* psi0, const real* f, const real* k2,
    const real* a, const real* b, const real* c,
    real* out, real* lap, int n, int spatial_n, int stage, real dt, real phase_rate) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    real ci = stage == STAGES ? (real)1 : c[stage];
    real sn, cs;
    phase(phase_rate * dt * ci * k2[{spatial_index}], &sn, &cs);
    real re = cs * psi0[2*k] - sn * psi0[2*k+1];
    real im = sn * psi0[2*k] + cs * psi0[2*k+1];
    for (int j = 0; j < stage; ++j) {{
        real weight = dt * (stage == STAGES ? b[j] : a[stage*STAGES+j]);
        if (weight == (real)0) continue;
        phase(phase_rate * dt * (ci-c[j]) * k2[{spatial_index}], &sn, &cs);
        real fjre = f[2*(j*n+k)], fjim = f[2*(j*n+k)+1];
        re += weight * (cs*fjre - sn*fjim);
        im += weight * (sn*fjre + cs*fjim);
    }}
    real normalization = stage == STAGES ? (real)1 : (real)1 / (real)spatial_n;
    re *= normalization;
    im *= normalization;
    out[2*k] = re;
    out[2*k+1] = im;
#if ETA
    if (stage != STAGES) {{
        lap[2*k] = -k2[{spatial_index}] * re;
        lap[2*k+1] = -k2[{spatial_index}] * im;
    }}
#endif
}}
extern "C" __global__ void stage_rhs(
    const real* psi, const real* lap,
    const real* res0, const real* dr, const real* a,
    const real* vr, const real* vi, const real* pump,
    const real* scales, int scale_index,
    real* rhs, real* out_dr, int n, int spatial_n, int pump_n, int stage, real dt,
    real inv_hbar, real hbar_over_2m, real gC, real gR, real gI,
    real R, real gammaC, real gammaR, real gammaI, real gammaA,
    real ria, real rai, real kappa, real eta) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    real active = res0[k];
    real inactive = MODEL == 0 ? (real)0 : res0[n+k];
    for (int j = 0; j < stage; ++j) {{
        real weight = dt * a[stage*STAGES+j];
        if (weight == (real)0) continue;
        active += weight * dr[(j*FIELDS)*n+k];
#if MODEL != 0
        inactive += weight * dr[(j*FIELDS+1)*n+k];
#endif
    }}
    real pre = psi[2*k], pim = psi[2*k+1];
    real rho = pre*pre + pim*pim;
    real energy = vr[{spatial_index}] + gC*rho + gR*active + gI*inactive;
    real gain = ((R*active-gammaC)*(real)0.5 + vi[{spatial_index}]*inv_hbar);
    real out_re = gain*pre + energy*inv_hbar*pim;
    real out_im = gain*pim - energy*inv_hbar*pre;
#if ETA
    real relax = eta*active*hbar_over_2m;
    out_re += relax*lap[2*k];
    out_im += relax*lap[2*k+1];
#endif
    rhs[2*k] = out_re;
    rhs[2*k+1] = out_im;
    real source = scales[scale_index] * pump[{pump_index}];
#if MODEL == 0
    out_dr[(stage*FIELDS)*n+k] = source - (gammaR+R*rho)*active;
#elif MODEL == 1
    out_dr[(stage*FIELDS)*n+k] = ria*inactive - (gammaA+rai+R*rho)*active;
    out_dr[(stage*FIELDS+1)*n+k] = source - (gammaI+ria)*inactive + rai*active;
#else
    real transfer = kappa*inactive*inactive;
    out_dr[(stage*FIELDS)*n+k] = transfer - (gammaR+R*rho)*active;
    out_dr[(stage*FIELDS+1)*n+k] = source - transfer - gammaI*inactive;
#endif
}}
extern "C" __global__ void finish_res(
    const real* res0, const real* dr, const real* b,
    real* out, int n, real dt) {{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n*FIELDS) return;
    int field = k/n, point = k%n;
    real val = res0[k];
    for (int j = 0; j < STAGES; ++j) {{
        if (b[j] != (real)0) val += dt*b[j]*dr[(j*FIELDS+field)*n+point];
    }}
    out[k] = val < (real)0 ? (real)0 : val;
}}
"""


def _stage_finish_source(source, physics_fields):
    signature = ("real ria, real rai, real kappa, real eta, int vr_n, int vi_n,"
                 " const real* batch_constants) {" if physics_fields else
                 "real ria, real rai, real kappa, real eta) {")
    source = source.replace(
        signature,
        signature.replace(") {", ", const real* b, real* res_out) {")
    )
    source = source.replace("    real active = res0[k];",
                            "    real active0 = res0[k];\n    real active = active0;")
    source = source.replace("    real inactive = MODEL == 0 ? (real)0 : res0[n+k];",
                            "    real inactive0 = MODEL == 0 ? (real)0 : res0[n+k];\n"
                            "    real inactive = inactive0;")
    source = source.replace("out_dr[(stage*FIELDS)*n+k] =", "real deriv_active =")
    source = source.replace("out_dr[(stage*FIELDS+1)*n+k] =", "real deriv_inactive =")
    before, after = source.split('extern "C" __global__ void finish_res(', 1)
    end = before.rfind("\n}")
    finish = """
    if (stage == STAGES-1) {
        for (int field = 0; field < FIELDS; ++field) {
            int point = field*n+k;
            real val = field == 0 ? active0 : inactive0;
            for (int j = 0; j < STAGES-1; ++j) {
                if (b[j] != (real)0)
                    val += dt*b[j]*dr[(j*FIELDS+field)*n+k];
            }
            real derivative = deriv_active;
#if MODEL != 0
            if (field == 1) derivative = deriv_inactive;
#endif
            if (b[STAGES-1] != (real)0) val += dt*b[STAGES-1]*derivative;
            res_out[point] = val < (real)0 ? (real)0 : val;
        }
    } else {
        out_dr[(stage*FIELDS)*n+k] = deriv_active;
#if MODEL != 0
        out_dr[(stage*FIELDS+1)*n+k] = deriv_inactive;
#endif
    }"""
    return before[:end] + finish + before[end:] + 'extern "C" __global__ void finish_res(' + after


def fused_source(real, model, eta, stages, fields, spatial_index, pump_index,
                 sincos, physics_fields=(), stage0_direct=False,
                 finish_in_stage=False):
    """Render the unchanged scalar source or a member-indexed physics variant."""
    source = _SOURCE.format(real=real, model=model, eta=eta, stages=stages,
                            fields=fields, spatial_index=spatial_index,
                            pump_index=pump_index, sincos=sincos)
    if model == 3:
        source = source.replace("    real active = res0[k];",
                                "    real active = (real)0;")
        source = source.replace("    real inactive = MODEL == 0 ? (real)0 : res0[n+k];",
                                "    real inactive = (real)0;")
        start = source.index("    for (int j = 0; j < stage; ++j)",
                             source.index('extern "C" __global__ void stage_rhs('))
        end = source.index("    real pre =", start)
        source = source[:start] + source[end:]
        start = source.index("    real source = scales[scale_index]")
        end = source.index('\n}', start)
        source = source[:start] + source[end:]
    if stage0_direct:
        source = source.replace(
            "    real pre = psi[2*k], pim = psi[2*k+1];",
            "    real pre = psi[2*k], pim = psi[2*k+1];\n"
            "    if (stage == 0) {\n"
            "        pre *= (real)1 / (real)spatial_n;\n"
            "        pim *= (real)1 / (real)spatial_n;\n"
            "    }",
        )
    if not physics_fields:
        return _stage_finish_source(source, False) if finish_in_stage else source
    loads = {name: index for index, name in enumerate(physics_fields)}
    source = source.replace(
        "real dt, real phase_rate) {",
        "real dt, real phase_rate, const real* batch_constants) {",
    ).replace(
        "real ria, real rai, real kappa, real eta) {",
        "real ria, real rai, real kappa, real eta, int vr_n, int vi_n,"
        " const real* batch_constants) {",
    )
    source = source.replace("vr[k % spatial_n]", "vr[k % vr_n]")
    source = source.replace("vi[k % spatial_n]", "vi[k % vi_n]")
    start, rest = source.split("    if (k >= n) return;", 1)
    combine, rest = rest.split('extern "C" __global__ void stage_rhs(', 1)
    if "phase_rate" in loads:
        combine = re.sub(r"\bphase_rate\b", "phase_rate_member", combine)
        combine = (f"\n    real phase_rate_member = batch_constants[{loads['phase_rate']}*(n/spatial_n)+k/spatial_n];"
                   + combine)
    source = start + "    if (k >= n) return;" + combine + 'extern "C" __global__ void stage_rhs(' + rest
    before_stage = source.split("    if (k >= n) return;", 2)
    prefix = "    if (k >= n) return;".join(before_stage[:2]) + "    if (k >= n) return;"
    rest = before_stage[2]
    stage, finish = rest.split('extern "C" __global__ void finish_res(', 1)
    names = tuple(name for name in physics_fields if name != "phase_rate")
    for name in names:
        stage = re.sub(rf"\b{name}\b", f"{name}_member", stage)
    stage = "".join(
        f"\n    real {name}_member = batch_constants[{loads[name]}*(n/spatial_n)+k/spatial_n];"
        for name in names
    ) + stage
    source = prefix + stage + 'extern "C" __global__ void finish_res(' + finish
    return _stage_finish_source(source, True) if finish_in_stage else source


def spinor_fused_source(real, model, eta, stages, fields, reservoir_components):
    """Render paired Lawson stages with the matrix exponential in combine_k."""
    source = r'''
typedef REAL real;
#define MODEL MODEL_ID
#define ETA ETA_ID
#define STAGES STAGE_COUNT
#define FIELDS FIELD_COUNT
#define RESCOMP RES_COUNT
__device__ __forceinline__ void spinor_apply(const real* matrix, int m,
    real tau, real ar, real ai, real br, real bi,
    real* xr, real* xi, real* yr, real* yi, int spatial_n) {
    real e0=matrix[m], e1=matrix[m+6*spatial_n];
    real z=(e0-e1)*(real)0.5, epsilon=(e0+e1)*(real)0.5;
    real ur=matrix[m+2*spatial_n], ui=matrix[m+2*spatial_n+1];
    real omega=sqrt(z*z+ur*ur+ui*ui);
    real theta=omega*tau;
    real s, c, ps, pc;
    sincos(theta, &s, &c);
    sincos(epsilon*tau, &ps, &pc);
    real factor=omega == (real)0 ? tau : s/omega;
    real vr=z*ar+ur*br-ui*bi;
    real vi=z*ai+ur*bi+ui*br;
    real wr=ur*ar+ui*ai-z*br;
    real wi=ur*ai-ui*ar-z*bi;
    real x0=c*ar+factor*vi, x1=c*ai-factor*vr;
    real y0=c*br+factor*wi, y1=c*bi-factor*wr;
    *xr=pc*x0+ps*x1; *xi=pc*x1-ps*x0;
    *yr=pc*y0+ps*y1; *yi=pc*y1-ps*y0;
}
extern "C" __global__ void combine_k(
    const real* psi0, const real* f, const real* k2,
    const real* a, const real* b, const real* c,
    real* out, real* lap, int n, int spatial_n, int stage, real dt, real phase_rate,
    const real* matrix, int matrix_members, const real* hbar) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k >= n) return;
    int member=k/spatial_n, cell=k%spatial_n;
    int p=2*(member*2*spatial_n+cell), q=p+2*spatial_n;
    int m=2*((matrix_members == 1 ? 0 : member)*4*spatial_n+cell);
    real ci=stage == STAGES ? (real)1 : c[stage];
    real xr, xi, yr, yi;
    spinor_apply(matrix,m,dt*ci/hbar[member],
        psi0[p],psi0[p+1],psi0[q],psi0[q+1],&xr,&xi,&yr,&yi,spatial_n);
    for (int j=0; j<stage; ++j) {
        real weight=dt*(stage == STAGES ? b[j] : a[stage*STAGES+j]);
        if (weight == (real)0) continue;
        int fp=p+2*j*2*n, fq=q+2*j*2*n;
        real fr, fi, gr, gi;
        spinor_apply(matrix,m,dt*(ci-c[j])/hbar[member],
            f[fp],f[fp+1],f[fq],f[fq+1],&fr,&fi,&gr,&gi,spatial_n);
        xr+=weight*fr; xi+=weight*fi;
        yr+=weight*gr; yi+=weight*gi;
    }
    real normalization=stage == STAGES ? (real)1 : (real)1/(real)spatial_n;
    xr*=normalization; xi*=normalization;
    yr*=normalization; yi*=normalization;
    out[p]=xr; out[p+1]=xi; out[q]=yr; out[q+1]=yi;
#if ETA
    if (stage != STAGES) {
        real factor=-k2[cell];
        lap[p]=factor*xr; lap[p+1]=factor*xi;
        lap[q]=factor*yr; lap[q+1]=factor*yi;
    }
#endif
}
extern "C" __global__ void stage_rhs(
    const real* psi, const real* lap,
    const real* res0, const real* dr, const real* a,
    const real* vr, const real* vi, const real* pump,
    const real* scales, int scale_index,
    real* rhs, real* out_dr, int n, int spatial_n, int pump_n, int stage, real dt,
    real inv_hbar, real hbar_over_2m, real gC, real gR, real gI,
    real R, real gammaC, real gammaR, real gammaI, real gammaA,
    real ria, real rai, real kappa, real eta,
    const real* constants, const real* weights, int vr_n, int vi_n) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k >= n) return;
    int member=k/spatial_n, cell=k%spatial_n;
    int res_n=n*RESCOMP;
    const real* q=constants+15*member;
    inv_hbar=q[0]; hbar_over_2m=q[1]; gC=q[2];
    real alpha2=q[3]; gR=q[4]; gI=q[5]; R=q[6]; gammaC=q[7];
    gammaR=q[8]; gammaI=q[9]; gammaA=q[10]; ria=q[11];
    rai=q[12]; kappa=q[13]; eta=q[14];
    real rho[2];
    for (int component=0; component<2; ++component) {
        int point=(member*2+component)*spatial_n+cell;
        real pre=psi[2*point], pim=psi[2*point+1];
        rho[component]=pre*pre+pim*pim;
    }
    for (int component=0; component<2; ++component) {
        int point=(member*2+component)*spatial_n+cell;
        int r=(member*RESCOMP+(RESCOMP == 1 ? 0 : component))*spatial_n+cell;
        real active=MODEL == 3 ? (real)0 : res0[r];
        real inactive=(MODEL == 0 || MODEL == 3) ? (real)0 : res0[res_n+r];
#if MODEL != 3
        for (int j=0; j<stage; ++j) {
            real weight=dt*a[stage*STAGES+j];
            if (weight == (real)0) continue;
            active+=weight*dr[(j*FIELDS)*res_n+r];
#if MODEL != 0
            inactive+=weight*dr[(j*FIELDS+1)*res_n+r];
#endif
        }
#endif
        real pre=psi[2*point], pim=psi[2*point+1];
        int pv=vr_n == spatial_n ? cell :
            (vr_n == 2*spatial_n ? component*spatial_n+cell : point);
        int pi=vi_n == spatial_n ? cell :
            (vi_n == 2*spatial_n ? component*spatial_n+cell : point);
        real energy=vr[pv]+gC*rho[component]+alpha2*rho[1-component]
            +gR*active+gI*inactive;
        real gain=(R*active-gammaC)*(real)0.5+vi[pi]*inv_hbar;
        real out_re=gain*pre+energy*inv_hbar*pim;
        real out_im=gain*pim-energy*inv_hbar*pre;
#if ETA
        real relax=eta*active*hbar_over_2m;
        out_re+=relax*lap[2*point];
        out_im+=relax*lap[2*point+1];
#endif
        rhs[2*point]=out_re; rhs[2*point+1]=out_im;
#if MODEL != 3
        if (RESCOMP == 2 || component == 0) {
            int pump_index=pump_n == spatial_n ? cell : k;
            real source=scales[scale_index]*pump[pump_index]
                *(RESCOMP == 1 ? (real)1 : weights[2*member+component]);
            real density=RESCOMP == 1 ? rho[0]+rho[1] : rho[component];
#if MODEL == 0
            out_dr[(stage*FIELDS)*res_n+r]=source-(gammaR+R*density)*active;
#elif MODEL == 1
            out_dr[(stage*FIELDS)*res_n+r]=ria*inactive-(gammaA+rai+R*density)*active;
            out_dr[(stage*FIELDS+1)*res_n+r]=source-(gammaI+ria)*inactive+rai*active;
#else
            real transfer=kappa*inactive*inactive;
            out_dr[(stage*FIELDS)*res_n+r]=transfer-(gammaR+R*density)*active;
            out_dr[(stage*FIELDS+1)*res_n+r]=source-transfer-gammaI*inactive;
#endif
        }
#endif
    }
}
extern "C" __global__ void finish_res(
    const real* res0, const real* dr, const real* b,
    real* out, int n, real dt) {
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    int res_n=n*RESCOMP;
    if (k >= res_n*FIELDS) return;
    int field=k/res_n, point=k%res_n;
    real val=res0[k];
    for (int j=0; j<STAGES; ++j) {
        if (b[j] != (real)0) val+=dt*b[j]*dr[(j*FIELDS+field)*res_n+point];
    }
    out[k]=val < (real)0 ? (real)0 : val;
}
'''
    return (source.replace("REAL", real).replace("MODEL_ID", str(model))
            .replace("ETA_ID", str(eta)).replace("STAGE_COUNT", str(stages))
            .replace("FIELD_COUNT", str(fields)).replace("RES_COUNT", str(reservoir_components)))


def spinor_feature_source(source, real, terms, varying_potential, drive):
    """Add local potential, coherent drive, and term snippets to paired stages."""
    source = source.replace("rho[2]", "densities[2]")
    source = source.replace("rho[component]", "densities[component]")
    source = source.replace("rho[1-component]", "densities[1-component]")
    source = source.replace("rho[0]+rho[1]", "densities[0]+densities[1]")
    declarations = []
    if varying_potential:
        declarations.extend(("const real* cap_re", "const real* cap_im"))
    if drive:
        declarations.extend(("const real* drive_re", "const real* drive_im",
                             "const real* drive_energy"))
    if terms:
        declarations.extend(("const real* x_coords", "const real* y_coords"))
    loads = []
    energy_terms = []
    rate_terms = []
    for number, term in enumerate(terms):
        expression = term.cuda.expression
        for ctype, name, *layout in term.cuda.arguments:
            kind = layout[0] if layout else "raw"
            identifier = f"term_{number}_{name}"
            concrete = ctype.replace("{REAL}", real)
            if kind == "raw":
                declarations.append(f"{concrete} {identifier}")
            else:
                declarations.append(f"{concrete} {identifier}_arg")
                item_type = concrete.replace("const", "").replace("*", "").strip()
                index = "cell" if kind == "shared" else "point"
                loads.append(f"        {item_type} {identifier} = {identifier}_arg[{index}];")
            expression = re.sub(rf"\b{name}\b", identifier, expression)
        (energy_terms if term.category == "energy" else rate_terms).append(
            f"({expression})")
    declarations.extend(("const real* feature_values", "int feature_index",
                         "int feature_members", "int feature_vr_n", "int feature_vi_n"))
    signature = "const real* constants, const real* weights, int vr_n, int vi_n) {"
    source = source.replace(signature, signature.replace(") {", ",\n    " +
                            ", ".join(declarations) + ") {"))
    local = ("        int feature_offset=(feature_index*feature_members+member)*3;\n"
             "        real potential_scale=feature_values[feature_offset];\n"
             "        real drive_scale=feature_values[feature_offset+1];\n"
             "        real t_stage=feature_values[feature_offset+2];\n"
             "        real rho=densities[component], n_active=active, n_inactive=inactive;\n"
             "        real psi_re=pre, psi_im=pim;\n")
    if terms:
        local += "        real x=x_coords[cell], y=y_coords[cell];\n"
        local += "\n".join(loads) + ("\n" if loads else "")
    local += ("        real extra_energy=" +
              ("+".join(energy_terms) or "(real)0") + ";\n"
              "        real extra_rate=" +
              ("+".join(rate_terms) or "(real)0") + ";\n")
    anchor = "        real energy=vr[pv]+gC*densities[component]"
    source = source.replace(anchor, local + anchor)
    if varying_potential:
        source = source.replace("vr[pv]+gC", "(vr[pv]*potential_scale+cap_re[point])+gC")
        source = source.replace("vi[pi]*inv_hbar", "(vi[pi]*potential_scale+cap_im[point])*inv_hbar")
    if drive:
        if varying_potential:
            source = source.replace("(vr[pv]*potential_scale+cap_re[point])+gC",
                "(vr[pv]*potential_scale+cap_re[point]-drive_energy[point])+gC")
        else:
            source = source.replace("vr[pv]+gC", "(vr[pv]-drive_energy[point])+gC")
    source = source.replace("+gR*active+gI*inactive;",
                            "+gR*active+gI*inactive+extra_energy;")
    source = source.replace("+vi[pi]*inv_hbar;", "+vi[pi]*inv_hbar+extra_rate;")
    source = source.replace("+(vi[pi]*potential_scale+cap_im[point])*inv_hbar;",
                            "+(vi[pi]*potential_scale+cap_im[point])*inv_hbar+extra_rate;")
    if drive:
        source = source.replace("        rhs[2*point]=out_re;",
            "        out_re+=inv_hbar*drive_scale*drive_im[point];\n"
            "        out_im-=inv_hbar*drive_scale*drive_re[point];\n"
            "        rhs[2*point]=out_re;")
    return source


def _feature_source(source, real, terms, varying_potential, drive, batched):
    declarations = []
    if varying_potential:
        declarations.extend(("const real* cap_re", "const real* cap_im"))
    if drive:
        declarations.extend(("const real* drive_re", "const real* drive_im",
                             "const real* drive_energy"))
    if terms:
        declarations.extend(("const real* x_coords", "const real* y_coords"))
    loads = []
    for number, term in enumerate(terms):
        for ctype, name, *layout in term.cuda.arguments:
            kind = layout[0] if layout else "raw"
            kernel_name = f"term_{number}_{name}"
            concrete = ctype.replace("{REAL}", real)
            if kind == "raw":
                declarations.append(f"{concrete} {kernel_name}")
            else:
                declarations.append(f"{concrete} {kernel_name}_arg")
                item_type = concrete.replace("const", "").replace("*", "").strip()
                index = "cell" if kind == "shared" else "k"
                loads.append(f"    {item_type} {kernel_name} = {kernel_name}_arg[{index}];\n")
    declarations.extend(("const real* feature_values", "int feature_index",
                         "int feature_members", "int feature_vr_n",
                         "int feature_vi_n"))
    start = source.index('extern "C" __global__ void stage_rhs(')
    end = source.index(") {", start)
    source = source[:end] + ",\n    " + ", ".join(declarations) + source[end:]
    stage_start = source.index("    real energy =", start)
    local = ("    int cell = k % spatial_n, member = k / spatial_n;\n"
             "    int feature_offset = (feature_index*feature_members+member)*3;\n"
             "    real potential_scale = feature_values[feature_offset];\n"
             "    real drive_scale = feature_values[feature_offset+1];\n"
             "    real t_stage = feature_values[feature_offset+2];\n")
    if terms:
        local += ("    real n_active = active, n_inactive = inactive;\n"
                  "    real x = x_coords[cell], y = y_coords[cell];\n")
        local += "".join(loads)
    source = source[:stage_start] + local + source[stage_start:]
    energy_extras = []
    rate_extras = []
    for number, term in enumerate(terms):
        expression = term.cuda.expression
        for argument in term.cuda.arguments:
            expression = re.sub(rf"\b{argument[1]}\b", f"term_{number}_{argument[1]}", expression)
        (energy_extras if term.category == "energy" else rate_extras).append(f"({expression})")
    energy_line = re.search(r"(?m)^    real energy = (.*);$", source)
    gain_line = re.search(r"(?m)^    real gain = (.*);$", source)
    energy = energy_line.group(1)
    gain = gain_line.group(1)
    if varying_potential:
        energy = re.sub(r"vr\[[^]]+\]",
                        "(vr[k % feature_vr_n]*potential_scale+cap_re[k])", energy)
        gain = re.sub(r"vi\[[^]]+\]",
                      "(vi[k % feature_vi_n]*potential_scale+cap_im[k])", gain)
    if drive:
        energy += " - drive_energy[k]"
    if energy_extras:
        energy += " + " + " + ".join(energy_extras)
    if rate_extras:
        gain += " + " + " + ".join(rate_extras)
    source = source[:energy_line.start()] + f"    real energy = {energy};" + source[energy_line.end():]
    gain_line = re.search(r"(?m)^    real gain = (.*);$", source)
    source = source[:gain_line.start()] + f"    real gain = {gain};" + source[gain_line.end():]
    if drive:
        anchor = "    rhs[2*k] = out_re;"
        source = source.replace(anchor,
            "    out_re += inv_hbar*drive_scale*drive_im[k];\n"
            "    out_im -= inv_hbar*drive_scale*drive_re[k];\n" + anchor, 1)
    return source


class FusedIFRK:
    """Own fixed tableau coefficients, stage storage, and cuFFT plans."""

    def __init__(self, solver, grid):
        from cupyx.scipy import fft

        self.solver = solver
        self.model = solver._model_features
        self.spinor = solver._spinor
        self._has_features = bool(self.model.terms or self.model.drive is not None or
                                  self.model.time_dependent_potential)
        for term in self.model.terms:
            if term.cuda is None:
                raise ValueError(f"Term {type(term).__name__} has no CUDA snippet")
        self._noise = self.model.noise if self.model.noise is not None and self.model.noise.active else None
        self.xp = solver.xp
        self.fft = fft
        self.ndim = grid.ndim
        self._fft_forward = fft.fft if grid.ndim == 1 else fft.fft2
        self.shape = (1, 1, *grid.shape)
        self._batched = False
        self.spatial_n = int(np.prod(grid.shape))
        self.n = self.spatial_n
        self.s = solver.tableau.stages
        self.real = solver._real_dtype
        self.complex = solver._complex_dtype
        self.scalar = solver._scalar
        self.k2 = self.xp.ascontiguousarray(solver._k_squared).ravel()
        if self.spinor:
            self.res_components = self.model.reservoir_components
            self._spinor_matrix = self.xp.ascontiguousarray(
                solver._spinor_matrix, dtype=self.complex)
            members = solver.batch_physics.size or 1
            physics_names = ("hbar", "g_C", "alpha_2", "g_R", "g_I", "R",
                             "gamma_C", "gamma_R", "gamma_I", "gamma_A",
                             "R_IA", "R_AI", "kappa", "kinetic_relaxation_eta", "m_eff")
            host = {name: np.asarray(solver.batch_physics.host.get(
                name, np.full(members, getattr(solver.config.physics, name))),
                dtype=self.scalar) for name in physics_names}
            self._spinor_hbar = self.xp.asarray(host["hbar"])
            constants = (1 / host["hbar"], host["hbar"] / (2 * host["m_eff"]),
                         host["g_C"], host["alpha_2"], host["g_R"], host["g_I"],
                         host["R"], host["gamma_C"], host["gamma_R"], host["gamma_I"],
                         host["gamma_A"], host["R_IA"], host["R_AI"], host["kappa"],
                         host["kinetic_relaxation_eta"])
            self._spinor_constants = self.xp.asarray(np.stack(constants, axis=1).ravel())
            self._spinor_weights = (self.xp.ascontiguousarray(
                self.model.pump_weights, dtype=self.real).ravel()
                                    if self.model.pump_weights is not None else
                                    self.xp.ones(2 * members, dtype=self.real))
        self.a = self.xp.asarray(solver._a, dtype=self.real).ravel()
        self.b = self.xp.asarray(solver._b, dtype=self.real)
        self.c = self.xp.asarray(solver._c, dtype=self.real)
        self.unique_c = tuple(dict.fromkeys(solver._c))
        self.c_index = tuple(self.unique_c.index(c) for c in solver._c)
        self.dt = self.scalar(solver.config.solver.dt)
        # Host-side times use the configured dt, not its kernel-precision copy.
        self.host_dt = float(solver.config.solver.dt)
        physics = solver.batch_physics
        self.p = solver.config.physics
        self.rate = self.scalar(-self.p.hbar / (2.0 * self.p.m_eff))
        self.eta = physics.any_nonzero("kinetic_relaxation_eta")
        values = {}
        if "hbar" in physics.host or "m_eff" in physics.host:
            hbar = physics.host.get("hbar", np.full(physics.size, self.p.hbar))
            mass = physics.host.get("m_eff", np.full(physics.size, self.p.m_eff))
            values["phase_rate"] = -hbar / (2.0 * mass)
            values["hbar_over_2m"] = hbar / (2.0 * mass)
        if "hbar" in physics.host:
            values["inv_hbar"] = 1.0 / physics.host["hbar"]
        direct = {"g_C": "gC", "g_R": "gR", "g_I": "gI", "R": "R",
                  "gamma_C": "gammaC", "gamma_R": "gammaR", "gamma_I": "gammaI",
                  "gamma_A": "gammaA", "R_IA": "ria", "R_AI": "rai",
                  "kappa": "kappa", "kinetic_relaxation_eta": "eta"}
        for name, kernel_name in direct.items():
            if name in physics.host:
                values[kernel_name] = physics.host[name]
        self._physics_fields = tuple(values)
        self._physics_device = (self.xp.asarray(np.stack(tuple(values.values())), dtype=self.real).ravel()
                                if values else None)
        self._physics_arg = (self._physics_device,) if values else ()
        self._kernels = {}
        self._plan = None
        self._buffers = None
        self._stream = self.xp.cuda.Stream(non_blocking=True)
        self._graph_cache = {}
        self._graph_signature = None
        self._graphs_failed = False
        self._potential_cached = None
        self._zero_pump = self.xp.zeros(self.shape, dtype=self.real)
        self._zero_res = self.xp.zeros(self.shape, dtype=self.real)
        self._feature_static = ()
        self._feature_buffer = None
        self._noise_step_base = (self.xp.zeros(1, dtype=self.xp.uint64)
                                 if self._noise is not None else None)
        if self._has_features:
            self._prepare_features()

    def _model(self, reservoir):
        from polarism.reservoir.single_reservoir import SingleReservoir
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.no_reservoir import NoReservoir

        models = {SingleReservoir: (0, 1), DoubleReservoir: (1, 2),
                  QuadraticDoubleReservoir: (2, 2), NoReservoir: (3, 0)}
        if type(reservoir) not in models:
            return None
        if type(reservoir) is not NoReservoir and any(reservoir.batch_physics.any_nonzero(name) for name in
               ("reservoir_diffusion_R", "reservoir_diffusion_A", "reservoir_diffusion_I")):
            return None
        return models[type(reservoir)]

    def eligible(self, reservoir):
        return self._model(reservoir) is not None

    def _array(self, value, dtype):
        return self.xp.ascontiguousarray(self.xp.asarray(value, dtype=dtype))

    def _potential(self, potential):
        if potential is not self._potential_cached:
            self._graph_cache.clear()
            self._potential_cached = None
            self.vr = self.vi = None
            value = (self.xp.broadcast_to(potential, self.shape)
                     if getattr(self, "spinor", False) else potential)
            vr = self._array(self.xp.real(value), self.real).ravel()
            vi = (self._array(self.xp.imag(value), self.real).ravel()
                  if self.xp.iscomplexobj(potential) else self._zero_pump.ravel())
            self.vr, self.vi = vr, vi
            self._potential_cached = potential

    def _prepare_features(self):
        xp = self.xp
        model = self.model
        arrays = []
        if model.time_dependent_potential:
            cap = xp.broadcast_to(xp.asarray(model.cap), self.shape)
            arrays.extend((self._array(xp.real(cap), self.real).ravel(),
                           self._array(xp.imag(cap), self.real).ravel()))
        if model.drive is not None:
            drive = xp.broadcast_to(xp.asarray(model.drive.spatial), self.shape)
            energy = xp.broadcast_to(xp.asarray(model.drive.energy), self.shape)
            arrays.extend((self._array(xp.real(drive), self.real).ravel(),
                           self._array(xp.imag(drive), self.real).ravel(),
                           self._array(energy, self.real).ravel()))
        if model.terms:
            arrays.extend((self._array(model.grid.X, self.real).ravel(),
                           self._array(model.grid.Y, self.real).ravel()
                           if model.grid.ndim == 2 else xp.zeros(self.spatial_n, self.real)))
        for term in model.terms:
            values = term.cuda_values()
            for ctype, name, *layout in term.cuda.arguments:
                value = values[name]
                if "*" in ctype:
                    dtype = (xp.uint64 if "long long" in ctype else
                             xp.uint32 if "unsigned int" in ctype else
                             xp.int32 if "int" in ctype else
                             xp.float64 if "double" in ctype else
                             xp.float32 if "float" in ctype else self.real)
                    arrays.append(self._array(value, dtype).ravel())
                elif "int" in ctype:
                    arrays.append(np.int32(value))
                else:
                    arrays.append(np.float64(value) if "double" in ctype else
                                  np.float32(value) if "float" in ctype else self.scalar(value))
        self._feature_static = tuple(arrays)
        self._feature_buffer = xp.empty(64 * len(self.unique_c) * (self.n // self.spatial_n) * 3,
                                        self.real)

    def _feature_values(self, times, stream=None):
        members = self.n // self.spatial_n
        values = np.empty((len(times), members, 3), dtype=self.scalar)
        def sample(value):
            host = np.asarray(value.get() if hasattr(value, "get") else value)
            return np.broadcast_to(host.reshape(-1), (members,))
        for index, time in enumerate(times):
            values[index, :, 0] = (sample(self.model.potential_envelope(time))
                                   if self.model.time_dependent_potential else 1)
            values[index, :, 1] = (sample(self.model.drive.envelope(time))
                                   if self.model.drive is not None else 0)
            values[index, :, 2] = time
        self._feature_buffer[:values.size].set(values.ravel(), stream=stream)
        return values

    def _noise_kernel(self, graph):
        if self.spinor:
            key = "_spinor_noise_graph" if graph else "_spinor_noise_regular"
            if not hasattr(self, key):
                from polarism.noise.philox import cuda_source
                source = cuda_source(self.solver._precision)
                source = source.replace("int member = j / spatial_n;",
                    "int member = j / (2*spatial_n); int component = (j / spatial_n) % 2;")
                source = source.replace("unsigned int d = 1U;",
                                        "unsigned int d = 1U + (unsigned int)component;")
                source = source.replace("0x1p-24f", "5.960464477539063e-8f")
                source = source.replace("0x1p-53", "1.1102230246251565e-16")
                source = source.replace("active[j]",
                    f"active[(member*{self.res_components}+(component*({self.res_components}==2)))*spatial_n+j%spatial_n]")
                if graph:
                    source = source.replace(
                        "unsigned long long step, int spatial_n, int total_n,",
                        "const unsigned long long* step_base, unsigned int step_offset, int spatial_n, int total_n,")
                    source = source.replace("    unsigned int b = (unsigned int)step;",
                        "    unsigned long long step = step_base[0] + step_offset;\n"
                        "    unsigned int b = (unsigned int)step;")
                setattr(self, key, self.xp.RawKernel(
                    source, f"add_wigner_{self.solver._precision}"))
            return getattr(self, key)
        if not graph:
            return self._noise._kernel
        if not hasattr(self, "_noise_graph"):
            from polarism.noise.philox import cuda_source
            source = cuda_source(self.solver._precision).replace(
                "unsigned long long step, int spatial_n, int total_n,",
                "const unsigned long long* step_base, unsigned int step_offset, int spatial_n, int total_n,")
            source = source.replace("    unsigned int b = (unsigned int)step;",
                                    "    unsigned long long step = step_base[0] + step_offset;\n"
                                    "    unsigned int b = (unsigned int)step;")
            self._noise_graph = self.xp.RawKernel(source, f"add_wigner_{self.solver._precision}")
        return self._noise_graph

    def _ensure(self, model):
        xp = self.xp
        from cupy.cuda import cufft
        model_id, fields = model
        if (self._buffers is not None and self._model_id == model_id
                and self._buffer_shape == self.shape):
            return
        self._graph_cache.clear()
        self._graph_signature = None
        self._pump_combiner = None
        self._plan = None
        self._graph_scales = None
        self._buffers = None
        self._model_id = None
        self._buffer_shape = None
        stage0_direct = (self._batched and not self.eta
                         and self.solver._c[0] == 0
                         and all(value == 0 for value in self.solver._a[0]))
        finish_in_stage = self._batched and model_id != 3 and not self.spinor
        if self.spinor:
            stage0_direct = False
        spec = tuple(xp.empty(self.shape, self.complex) for _ in range(2))
        res_shape = ((fields, self.shape[0], self.res_components, *self.shape[2:])
                     if self.spinor else (fields, *self.shape))
        res = tuple(xp.empty(res_shape, self.real) for _ in range(2))
        f = xp.empty((self.s, *self.shape), self.complex)
        dr = xp.empty((self.s, *res_shape), self.real)
        stage_k = xp.empty(self.shape, self.complex)
        stage_psi = xp.empty_like(stage_k)
        lap_k = xp.empty_like(stage_k)
        lap_psi = xp.empty_like(stage_k)
        rhs = xp.empty_like(stage_k)
        source = (spinor_fused_source(
            "float" if self.real == xp.float32 else "double", model_id, int(self.eta),
            self.s, fields, self.res_components) if self.spinor else fused_source(
            real="float" if self.real == xp.float32 else "double",
                              model=model_id, eta=int(self.eta), stages=self.s,
                              fields=fields,
                              spatial_index="k" if not self._batched else "k % spatial_n",
                              pump_index="k" if not self._batched else "k % pump_n",
                              sincos="sincosf" if self.real == xp.float32 else "sincos",
                              physics_fields=self._physics_fields,
                              stage0_direct=stage0_direct,
                              finish_in_stage=finish_in_stage))
        if self._has_features:
            real_type = "float" if self.real == xp.float32 else "double"
            source = (spinor_feature_source(
                source, real_type, self.model.terms,
                self.model.time_dependent_potential, self.model.drive is not None)
                if self.spinor else _feature_source(
                    source, real_type, self.model.terms,
                    self.model.time_dependent_potential,
                    self.model.drive is not None, self._batched))
        module = xp.RawModule(code=source, options=("--std=c++11",))
        kernels = {name: module.get_function(name)
                   for name in ("combine_k", "stage_rhs", "finish_res")}
        plan = self.fft.get_fft_plan(stage_k, axes=tuple(range(-self.ndim, 0)))
        graph_scales = xp.empty((64, len(self.unique_c)), self.real)
        # Warm cuFFT before graph capture. Run on the current stream: warming the
        # plan on the private non-blocking stream left later FFTs racing the
        # default-stream kernels, giving wrong results in some solver instances.
        stage_k.fill(0)
        plan.fft(stage_k, stage_psi, cufft.CUFFT_INVERSE)
        plan.fft(stage_psi, f[0], cufft.CUFFT_FORWARD)
        xp.cuda.get_current_stream().synchronize()
        self._buffers = (spec, res, f, dr, stage_k, stage_psi, lap_k, lap_psi, rhs)
        self._kernels = kernels
        self._plan = plan
        self._graph_scales = graph_scales
        self._stage0_direct = stage0_direct
        self._finish_in_stage = finish_in_stage
        self._model_id = model_id
        self._buffer_shape = self.shape

    def _pump(self, pump, t):
        term = pump.separable_term() if hasattr(pump, "separable_term") else None
        terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
        active_terms = terms if terms is not None else (term,) if term is not None else ()
        no_lasers = not getattr(pump, "lasers", [True])
        if self._batched and any(np.ndim(coefficient(t)) != 0
                                        for _, coefficient in active_terms):
            from polarism.solver.cuda_pump import CudaPumpCombiner
            if getattr(self, "_pump_combiner", None) is None:
                self._pump_combiner = CudaPumpCombiner(self.xp, self.real, self.n)
            combiner = self._pump_combiner
            combiner.prepare(active_terms, len(self.unique_c), len(self.unique_c),
                             shape=self.shape)
            times = [t + c * self.host_dt for c in self.unique_c]
            stream = self.xp.cuda.get_current_stream()
            combiner.coefficients(active_terms, times, stream)
            stream.synchronize()
            samples = [combiner.combine(i, i) for i in range(len(times))]
            return ([samples[i] for i in self.c_index], [1] * len(times), None, None)
        if term is not None:
            envelope, coefficient = term
            field = self._array(envelope, self.real).ravel()
            scales = [coefficient(t + c * self.host_dt) for c in self.unique_c]
            return [field] * self.s, scales, field, coefficient
        if no_lasers:
            return ([self._zero_pump.ravel()] * self.s,
                    [1] * len(self.unique_c), self._zero_pump.ravel(), lambda time: 1)
        samples = [self._array(evaluate_pump(pump, t + c * self.host_dt), self.real).ravel()
                   for c in self.unique_c]
        return [samples[i] for i in self.c_index], [1] * len(samples), None, None

    def _run_into(self, spectrum_in, res_in, spectrum_out, res_out,
                  pumps, scales, scale_base, noise_step=None, graph_noise=False):
        from cupy.cuda import cufft

        _, _, f, dr, stage_k, stage_psi, lap_k, lap_psi, rhs = self._buffers
        launch = ((self.n + 255) // 256,)
        block = (256,)
        scalar = self.scalar
        p = self.p
        finish_args = ((self.b, res_out.ravel()) if self._finish_in_stage else ())
        combine_extra = ((self._spinor_matrix.view(self.real).ravel(),
                          np.int32(self._spinor_matrix.shape[0]), self._spinor_hbar)
                         if self.spinor else self._physics_arg)
        stage_extra = ((self._spinor_constants, self._spinor_weights,
                        np.int32(self.vr.size), np.int32(self.vi.size))
                       if self.spinor else
                       ((np.int32(self.vr.size), np.int32(self.vi.size)) + self._physics_arg
                        if self._physics_fields else ()))
        for i in range(self.s):
            if i == 0 and self._stage0_direct:
                self._plan.fft(spectrum_in, stage_psi, cufft.CUFFT_INVERSE)
            else:
                self._kernels["combine_k"](launch, block, (
                    spectrum_in.view(self.real).ravel(), f.view(self.real).ravel(), self.k2,
                    self.a, self.b, self.c, stage_k.view(self.real).ravel(),
                    lap_k.view(self.real).ravel(), np.int32(self.n), np.int32(self.spatial_n), np.int32(i),
                    self.dt, self.rate) + combine_extra)
                self._plan.fft(stage_k, stage_psi, cufft.CUFFT_INVERSE)
            if self.eta:
                self._plan.fft(lap_k, lap_psi, cufft.CUFFT_INVERSE)
            self._kernels["stage_rhs"](launch, block, (
                stage_psi.view(self.real).ravel(), lap_psi.view(self.real).ravel(),
                res_in.ravel(), dr.ravel(), self.a, self.vr, self.vi, pumps[i],
                scales, np.int32(scale_base + self.c_index[i]),
                rhs.view(self.real).ravel(), dr.ravel(), np.int32(self.n),
                np.int32(self.spatial_n), np.int32(pumps[i].size), np.int32(i), self.dt,
                scalar(1 / p.hbar), scalar(p.hbar / (2 * p.m_eff)),
                scalar(p.g_C), scalar(p.g_R), scalar(p.g_I), scalar(p.R),
                scalar(p.gamma_C), scalar(p.gamma_R), scalar(p.gamma_I),
                scalar(p.gamma_A), scalar(p.R_IA), scalar(p.R_AI),
                scalar(p.kappa), scalar(p.kinetic_relaxation_eta))
                + stage_extra + finish_args
                + ((*self._feature_static, self._feature_buffer,
                    np.int32(scale_base + self.c_index[i]),
                    np.int32(self.n // self.spatial_n), np.int32(self.vr.size),
                    np.int32(self.vi.size)) if self._has_features else ()))
            self._plan.fft(rhs, f[i], cufft.CUFFT_FORWARD)
        self._kernels["combine_k"](launch, block, (
            spectrum_in.view(self.real).ravel(), f.view(self.real).ravel(), self.k2,
            self.a, self.b, self.c, spectrum_out.view(self.real).ravel(),
            lap_k.view(self.real).ravel(), np.int32(self.n), np.int32(self.spatial_n), np.int32(self.s),
            self.dt, self.rate) + combine_extra)
        if not self._finish_in_stage and res_in.shape[0]:
            reservoir_n = self.n * (self.res_components if self.spinor else 1)
            self._kernels["finish_res"](((reservoir_n*res_in.shape[0]+255)//256,), block, (
                res_in.ravel(), dr.ravel(), self.b, res_out.ravel(),
                np.int32(self.n), self.dt))
        if self._noise is not None and noise_step is not None:
            xp = self.xp
            xp.multiply(spectrum_out, self.scalar(1 / self.spatial_n), out=stage_k)
            self._plan.fft(stage_k, stage_psi, cufft.CUFFT_INVERSE)
            active = res_out[0] if res_out.shape[0] else self._zero_res
            noise_args = (stage_psi.view(self.real).ravel(), active.ravel(),
                          self._noise._device_seeds, self._noise._device_gamma,
                          self._noise._device_scattering, self._noise._device_strength)
            noise_args += ((self._noise_step_base, np.uint32(noise_step)) if graph_noise
                           else (np.uint64(noise_step),))
            noise_args += (np.int32(self.spatial_n),
                           np.int32(2*self.n if self.spinor else self.n),
                           self.scalar(self._noise.dt / (4.0 * self._noise.volume)))
            noise_launch = (((2*self.n+255)//256,) if self.spinor else launch)
            self._noise_kernel(graph_noise)(noise_launch, block, noise_args)
            self._plan.fft(stage_psi, spectrum_out, cufft.CUFFT_FORWARD)

    def _copy_in(self, state, reservoir):
        xp = self.xp
        spec, res = self._buffers[:2]
        spectrum = state.field_k
        if spectrum is None:
            psi = self._array(state.field, self.complex)
            spectrum = self._fft_forward(psi, plan=self._plan)
        xp.copyto(spec[0], spectrum)
        for target, source in zip(res[0], reservoir.get_state()):
            xp.copyto(target, source)

    def _copy_out(self, index, reservoir, state):
        spec, res = self._buffers[:2]
        state.field = None
        state.field_k = spec[index].copy()
        reservoir.set_state(tuple(field.copy() for field in res[index]))

    def _activate_shape(self, state):
        shape = (state.field_k.shape if state.field_k is not None else state.field.shape)
        if shape != self.shape or state.batched != self._batched:
            self.shape = shape
            self._batched = state.batched
            self.n = int(np.prod(shape)) // (2 if getattr(self, "spinor", False) else 1)
            self._graph_cache.clear()
            self._graph_signature = None
            self._pump_combiner = None
            self._buffers = None
            self._plan = None
            self._graph_scales = None
            self._model_id = None
            self._buffer_shape = None
            self._zero_pump = None
            self._zero_res = self.xp.zeros(self.shape, self.real)
            if getattr(self, "_has_features", False):
                self._prepare_features()
            self._potential_cached = None
            self.vr = self.vi = None
        if self._zero_pump is None:
            self._zero_pump = self.xp.zeros(self.shape, dtype=self.real)

    def step(self, physical_model, state):
        potential = (physical_model.potential_static if physical_model.time_dependent_potential
                     else physical_model.potential)
        pump = physical_model.pump
        reservoir = physical_model.reservoir
        boundary_condition = physical_model.boundary
        self._activate_shape(state)
        model = self._model(reservoir)
        self._ensure(model)
        self._potential(potential)
        pumps, factors, _, _ = self._pump(pump, state.t)
        scales = self.xp.asarray(factors, dtype=self.real)
        if self._has_features:
            self._feature_values([state.t + c * self.host_dt for c in self.unique_c])
        self._copy_in(state, reservoir)
        spec, res = self._buffers[:2]
        self._run_into(spec[0], res[0], spec[1], res[1], pumps, scales, 0)
        if not self.solver._can_skip_after_step(boundary_condition):
            state.field = None
            state.field_k = spec[1].copy()
            state.field = boundary_condition.after_step_action(state.field)
            state.field_k = self._fft_forward(state.field, plan=self._plan)
            reservoir.set_state(tuple(field.copy() for field in res[1]))
        else:
            self._copy_out(1, reservoir, state)

    def _graph(self, length, signature, pump_flat, combiner=None):
        if signature != self._graph_signature:
            self._graph_cache.clear()
            self._graph_signature = signature
        if length in self._graph_cache:
            return self._graph_cache[length]
        stream = self._stream
        spec, res = self._buffers[:2]
        pumps = [pump_flat] * self.s
        if self._noise is not None:
            self._noise_kernel(True).compile()
        stream.synchronize()
        with stream:
            stream.begin_capture()
            try:
                for k in range(length):
                    if combiner is not None:
                        samples = [combiner.combine(k * len(self.unique_c) + i, i)
                                   for i in range(len(self.unique_c))]
                        pumps = [samples[i] for i in self.c_index]
                    self._run_into(spec[k % 2], res[k % 2],
                                   spec[(k + 1) % 2], res[(k + 1) % 2],
                                   pumps, self._graph_scales.ravel(),
                                   k * len(self.unique_c), k, True)
                graph = stream.end_capture()
            except Exception:
                try:
                    stream.end_capture()
                except Exception:
                    pass
                raise
        if len(self._graph_cache) >= 4:
            self._graph_cache.pop(next(iter(self._graph_cache)))
        self._graph_cache[length] = graph
        return graph

    def advance(self, physical_model, state, n_steps):
        potential = (physical_model.potential_static if physical_model.time_dependent_potential
                     else physical_model.potential)
        pump = physical_model.pump
        reservoir = physical_model.reservoir
        boundary_condition = physical_model.boundary
        start_step = state.step
        self._activate_shape(state)
        model = self._model(reservoir)
        self._ensure(model)
        self._potential(potential)
        xp = self.xp
        xp.cuda.get_current_stream().synchronize()
        spec, res = self._buffers[:2]
        stream = self._stream
        done = 0
        with stream:
            self._copy_in(state, reservoir)
            terms = pump.separable_terms() if hasattr(pump, "separable_terms") else None
            if terms is None and hasattr(pump, "separable_term"):
                term = pump.separable_term()
                if term is not None:
                    terms = (term,)
            multi = terms is not None and (len(terms) > 1 or
                    (self._batched and len(terms) == 1 and
                     np.ndim(terms[0][1](state.t)) != 0))
            if multi:
                from polarism.solver.cuda_pump import CudaPumpCombiner
                if getattr(self, "_pump_combiner", None) is None:
                    self._pump_combiner = CudaPumpCombiner(xp, self.real, self.n)
                combiner = self._pump_combiner
                combiner.prepare(terms, len(self.unique_c), 64 * len(self.unique_c),
                                 shape=self.shape if self._batched else None)
                pump_flat = None
                coefficient = None
            else:
                combiner = None
                _, _, pump_flat, coefficient = self._pump(pump, state.t)
            graph_ok = (self.solver.config.solver.cuda_graphs and not self._graphs_failed
                        and (coefficient is not None or multi)
                        and self.solver._can_skip_after_step(boundary_condition))
            pending_hosts = []
            if graph_ok:
                nodes_per_step = ((4 + int(self.eta))*self.s + 2 + 4*int(self._noise is not None)
                                  - int(self._stage0_direct) - int(self._finish_in_stage)
                                  + (len(self.unique_c) if multi else 0))
                max_length = max(2, min(64, (1800 // nodes_per_step) // 2 * 2))
                signature = (self.shape, model, id(potential),
                             combiner.signature if multi else int(pump_flat.data.ptr))
                try:
                    while n_steps - done >= 2:
                        length = min(max_length, (n_steps-done)//2*2)
                        graph = self._graph(length, signature, pump_flat, combiner)
                        times = [state.t0+(start_step+done+k)*self.host_dt+c*self.host_dt
                                 for k in range(length) for c in self.unique_c]
                        if multi:
                            pending_hosts.append(combiner.coefficients(terms, times, stream))
                            host = self.scalar([1] * (length * len(self.unique_c)))
                        else:
                            host = self.scalar([coefficient(t) for t in times])
                        host = host.reshape(length, len(self.unique_c))
                        pending_hosts.append(host)
                        self._graph_scales[:length].set(host, stream=stream)
                        if self._has_features:
                            pending_hosts.append(self._feature_values(times, stream))
                        if self._noise is not None:
                            host_step = np.asarray([start_step + done], dtype=np.uint64)
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
                    print(f"ifrk-fft-cuda: CUDA graphs unavailable ({exc}); using regular launches.",
                          file=sys.stderr)
                    stream.synchronize()
                    self._copy_in(state, reservoir)
                    done = 0
            for k in range(done, n_steps):
                t = state.t0+(start_step+k)*self.host_dt
                if multi:
                    times = [t+c*self.host_dt for c in self.unique_c]
                    pending_hosts.append(combiner.coefficients(terms, times, stream))
                    samples = [combiner.combine(i, i) for i in range(len(times))]
                    stage_pumps = [samples[i] for i in self.c_index]
                    factors = [1] * len(times)
                else:
                    stage_pumps, factors, _, _ = self._pump(pump, t)
                scales = xp.asarray(factors, dtype=self.real)
                if self._has_features:
                    pending_hosts.append(self._feature_values(
                        [t + c * self.host_dt for c in self.unique_c], stream))
                parity = (k-done) % 2
                self._run_into(spec[parity], res[parity], spec[1-parity],
                               res[1-parity], stage_pumps, scales, 0, start_step + k)
                if len(pending_hosts) >= 32:
                    stream.synchronize()
                    pending_hosts.clear()
            index = (n_steps-done) % 2
            if self.solver._can_skip_after_step(boundary_condition):
                self._copy_out(index, reservoir, state)
            else:
                state.field = None
                state.field_k = spec[index].copy()
                state.field = boundary_condition.after_step_action(state.field)
                state.field_k = self._fft_forward(state.field, plan=self._plan)
                reservoir.set_state(tuple(field.copy() for field in res[index]))
            stream.synchronize()
        state.step = start_step + n_steps
        state.t = state.t0 + state.step * self.host_dt
