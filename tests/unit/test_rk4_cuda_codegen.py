"""Textual identity gate for all pre-4c rk4-cuda source variants."""

import hashlib
import json
import os
import re
from itertools import product
from pathlib import Path

from polarism.solver.rk4_cuda_solver import (
    build_flat_1d_sources,
    build_kernel_source_1d,
    build_tiled_1d_kernel_source,
    build_tiled_kernel_source,
    _fused_noise_source,
    _extend_cuda_source,
    _without_reservoir,
)
from polarism.model_terms import CudaSnippet


SNAPSHOT = Path("/tmp/claude-1000/-var-home-jakub-Dev-Polarism/982a41f5-ae42-44a0-8063-9768638126db/scratchpad/rk4_sources_before.json")
HASHES = {'flat:1:single': '658e0e71b6bf08d1987dae9a709112ab614e22316c144b6915048c5c95b300ab', 'tiled:1:single': '38c00bff59e920309be9e004b15f7515919301984b54f2ab365879ecbe44c92d', 'flat:1:double': '6bbf06d2759fe8b632f7b1e8e446dd482a4b1c9d94460a6127cf1283c59fb7ca', 'tiled:1:double': '4c9c22e505544e1706a3e97fbed4f641c107df60323bcb80b0eaa94e11b79a2e', 'flat:1:quadratic-double': '58bb3fece5ff22f27056857703359a0be8948f7c2450019c66cfefb91bfab23d', 'tiled:1:quadratic-double': '8ed97f09e1578cd0212a77a8526bd61e4050184ee1bd368e0dff3d627b5889ef', 'flat:2:single': 'e1fee7c79a3b6e6802e725cdf764600505c868d708440e9483b91fa043ebae52', 'tiled:2:single': '0eb5f95b818ba72dd475e2050dca1b5fd7d166f23fa4c4a030d157db58cc59ce', 'flat:2:double': '411b5181bc791e3826a82faa3828605e3b8999cad9bd222404bd42ef82692e89', 'tiled:2:double': '5c55d7e80d77c0b5b2b233fc62e94e434eec84cfb54c55abe1f4f67e8c7e59cb', 'flat:2:quadratic-double': 'b05ffd80c91cc1ab8cf497356de94e7ae3739e40fc383ac13721eee67bc1e85b', 'tiled:2:quadratic-double': '0d94b6a1e67e201c5453e28f4ffa5bda3cc7cd24f32d780c33366e83df4eceef', 'flat_graph:1:single': 'b72a6bf4bd0af45027966b1ad2aaec51cd7af3a4e2ec9c8f0dc4a638acdcd682', 'flat_graph:1:double': 'e44f035abe2fd3f7e3e44aca15297fa383aa50b08d161d00e11a82acb3cf5543', 'flat_graph:1:quadratic-double': '90a4deb009c99debddef815dd31d5405f5d8ed37a25c1f01f9dddb25fdf828a1', 'flat_graph:2:single': '23e8e793a2288c2e811eb1ea014966d3eaa8898743e2f3f36bc05569d5c5e361', 'flat_graph:2:double': '5903518e2f51fd527d3787910686d22604817cd5a1b2c1c0c24f250cce38fdba', 'flat_graph:2:quadratic-double': 'd2850e483cc0eab0b4cbb492cbbe32b2ccd1b2d7be04d310cc81edef8634de79'}
BATCH_FIELDS = (
    "hbar", "m_eff", "g_C", "g_R", "g_I", "R", "gamma_C", "gamma_R",
    "gamma_I", "gamma_A", "R_IA", "R_AI", "kappa",
    "kinetic_relaxation_eta", "reservoir_diffusion_R",
    "reservoir_diffusion_A", "reservoir_diffusion_I",
)


def sources():
    """Yield each existing source variant without creating a CUDA context."""
    for ndim, reservoir, boundary, real, batched in product(
        (1, 2), ("single", "double", "quadratic-double"),
        ("periodic", "closed-interval"), ("float", "double"), (False, True),
    ):
        field_sets = (((),) if not batched else
                      (((),) + tuple((field,) for field in BATCH_FIELDS) + (BATCH_FIELDS,)))
        laplacians = ("five-point",) if ndim == 1 else ("five-point", "isotropic-9pt")
        for fields, laplacian in product(field_sets, laplacians):
            stem = f"{ndim}:{reservoir}:{boundary}:{real}:{int(batched)}:{','.join(fields)}:{laplacian}"
            if ndim == 1:
                flat = build_flat_1d_sources(boundary, real, reservoir, batched, fields)
            else:
                flat = build_kernel_source_1d(boundary, real, reservoir, laplacian,
                                              batched=batched, physics_fields=fields)
            for stage, source in enumerate(flat):
                yield f"flat:{stem}:{stage}", source
                graph_source = source.replace(
                    f"const {real} pump_scale,",
                    f"const {real}* pump_scales, const int pump_index,",
                )
                graph_source = re.sub(r"\bpump_scale\b",
                                      "pump_scales[pump_index]", graph_source)
                yield f"flat_graph:{stem}:{stage}", graph_source
            for complex_potential, graph in product((False, True), repeat=2):
                if ndim == 1:
                    source = build_tiled_1d_kernel_source(
                        boundary, real, reservoir, 32, complex_potential, graph,
                        batched, fields,
                    )
                else:
                    source = build_tiled_kernel_source(
                        boundary, real, reservoir, laplacian, 16, 8,
                        complex_potential, graph, batched, fields,
                    )
                yield f"tiled:{stem}:{int(complex_potential)}:{int(graph)}", source


def test_existing_sources_are_textually_identical():
    """Compare every source to the baseline hash saved before the refactor."""
    rendered = dict(sources())
    if os.environ.get("RK4_SNAPSHOT_WRITE"):
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(rendered, separators=(",", ":")))
        return
    groups = {}
    for key, source in rendered.items():
        group = ":".join(key.split(":")[:3])
        groups.setdefault(group, {})[key] = source
    actual = {group: hashlib.sha256(json.dumps(items, sort_keys=True,
                      separators=(",", ":")).encode()).hexdigest()
              for group, items in groups.items()}
    assert actual == HASHES


def test_fused_noise_source_uses_final_density_and_graph_step():
    for reservoir, active in (("single", "nR_out[tid]"),
                              ("double", "nA_out[tid]"),
                              ("quadratic-double", "nR_out[tid]")):
        for real, batched, graph in product(("float", "double"),
                                            (False, True), (False, True)):
            source = build_kernel_source_1d("periodic", real, reservoir,
                                            batched=batched)[2]
            fused = _fused_noise_source(source, real, reservoir, batched, graph)
            assert "philox4x32(&a, &b, &c, &d" in fused
            assert "unsigned int d = 1U;" in fused
            assert active in fused
            assert ("noise_step_base[0] + noise_step_offset" in fused) == graph
            assert ("const int noise_member = member;" in fused) == batched


def test_plugin_expression_renders_fixed_local_names_and_arguments():
    snippet = CudaSnippet("-alpha * rho + field[cell]", (
        ("{REAL}", "alpha"), ("const {REAL}*", "field"),
    ))
    term = type("LocalTerm", (), {"category": "rate", "cuda": snippet})()
    source = _extend_cuda_source(
        build_kernel_source_1d("periodic", "float", "single")[0],
        "float", "single", False, (term,), True, True,
    )
    assert "const float pre = psi[2*tid];" in source
    assert "const float n_active = nR_val;" in source
    assert "const float* term_0_field" in source
    assert "float term_0_alpha" in source
    assert "(-term_0_alpha * rho + term_0_field[cell])" in source
    assert "V_re[tid] * potential_scale + V_cap_re[tid]" in source
    assert "drive_scale * drive_im[tid]" in source


def test_cuda_array_layouts_render_local_values():
    snippet = CudaSnippet("shared_mask + member_mask", (
        ("const {REAL}*", "shared_mask", "shared"),
        ("const {REAL}*", "member_mask", "batched"),
    ))
    term = type("LocalTerm", (), {"category": "energy", "cuda": snippet})()
    source = _extend_cuda_source(
        build_kernel_source_1d("periodic", "double", "single", batched=True)[0],
        "double", "single", True, (term,), False, False,
    )
    assert "term_0_shared_mask_arg[cell]" in source
    assert "term_0_member_mask_arg[tid]" in source
    assert "term_0_shared_mask + term_0_member_mask" in source


def test_none_sources_have_no_reservoir_state_or_equations():
    for ndim, batched, real in product((1, 2), (False, True), ("float", "double")):
        originals = (build_flat_1d_sources("periodic", real, "single", batched)
                     if ndim == 1 else build_kernel_source_1d(
                         "periodic", real, "single", batched=batched))
        for source in originals:
            rendered = _without_reservoir(source, real)
            assert "nR[" not in rendered
            assert "nR0[" not in rendered
            assert "k_nR[" not in rendered
            assert "dnR_dt[" not in rendered
            assert "nR_out[" not in rendered
            assert "lap_nR" not in rendered
            assert f"const {real} ({real})0" not in rendered
