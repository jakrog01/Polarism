"""Philox4x32-10 with a matching CUDA implementation."""
from __future__ import annotations

import numpy as np


def philox4x32(counter, key: int):
    """Return four uint32 words for each four-word counter and 64-bit key."""
    words = np.asarray(counter, dtype=np.uint32)
    if words.shape[-1] != 4:
        raise ValueError("Philox counters must have four uint32 words")
    c0, c1, c2, c3 = (words[..., i].copy() for i in range(4))
    k0 = np.uint32(key & 0xFFFFFFFF)
    k1 = np.uint32((key >> 32) & 0xFFFFFFFF)
    for round_index in range(10):
        p0 = np.uint64(c0) * np.uint64(0xD2511F53)
        p1 = np.uint64(c2) * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            np.uint32(p1 >> np.uint64(32)) ^ c1 ^ k0,
            np.uint32(p1),
            np.uint32(p0 >> np.uint64(32)) ^ c3 ^ k1,
            np.uint32(p0),
        )
        if round_index < 9:
            k0 = np.uint32((int(k0) + 0x9E3779B9) & 0xFFFFFFFF)
            k1 = np.uint32((int(k1) + 0xBB67AE85) & 0xFFFFFFFF)
    return np.stack((c0, c1, c2, c3), axis=-1)


def counter_words(cells, step: int, stream: int = 1):
    """Build canonical counters from spatial indices and the global step."""
    indices = np.asarray(cells, dtype=np.uint32)
    result = np.empty((*indices.shape, 4), dtype=np.uint32)
    result[..., 0] = indices
    result[..., 1] = np.uint32(step & 0xFFFFFFFF)
    result[..., 2] = np.uint32((step >> 32) & 0xFFFFFFFF)
    result[..., 3] = np.uint32(stream)
    return result


def normal_pair(words, precision: str):
    """Convert Philox words to two Box–Muller standard normals."""
    source = np.asarray(words, dtype=np.uint32)
    if precision == "single":
        u1 = ((source[..., 0] >> 8).astype(np.float32) + np.float32(1.0)) * np.float32(2.0 ** -24)
        u2 = ((source[..., 1] >> 8).astype(np.float32) + np.float32(1.0)) * np.float32(2.0 ** -24)
        radius = np.sqrt(np.float32(-2.0) * np.log(u1))
        angle = np.float32(2.0 * np.pi) * u2
    else:
        u1 = ((source[..., 0] >> 5).astype(np.float64) * 67108864.0
              + (source[..., 1] >> 6).astype(np.float64) + 1.0) * 2.0 ** -53
        u2 = ((source[..., 2] >> 5).astype(np.float64) * 67108864.0
              + (source[..., 3] >> 6).astype(np.float64) + 1.0) * 2.0 ** -53
        radius = np.sqrt(-2.0 * np.log(u1))
        angle = (2.0 * np.pi) * u2
    return radius * np.cos(angle), radius * np.sin(angle)


CUDA_SOURCE = r"""
extern "C" __device__ __forceinline__ void philox4x32(
    unsigned int* c0, unsigned int* c1, unsigned int* c2, unsigned int* c3,
    unsigned int k0, unsigned int k1) {
    for (int round = 0; round < 10; ++round) {
        unsigned long long p0 = (unsigned long long)(*c0) * 0xD2511F53ULL;
        unsigned long long p1 = (unsigned long long)(*c2) * 0xCD9E8D57ULL;
        unsigned int a = (unsigned int)(p1 >> 32) ^ (*c1) ^ k0;
        unsigned int b = (unsigned int)p1;
        unsigned int c = (unsigned int)(p0 >> 32) ^ (*c3) ^ k1;
        unsigned int d = (unsigned int)p0;
        *c0 = a; *c1 = b; *c2 = c; *c3 = d;
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
}
extern "C" __global__ void philox_words(
    const unsigned int* counters, unsigned int* output, unsigned long long key,
    int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    unsigned int a = counters[4*j], b = counters[4*j+1];
    unsigned int c = counters[4*j+2], d = counters[4*j+3];
    philox4x32(&a, &b, &c, &d, (unsigned int)key, (unsigned int)(key >> 32));
    output[4*j] = a; output[4*j+1] = b;
    output[4*j+2] = c; output[4*j+3] = d;
}
extern "C" __global__ void add_wigner_{suffix}(
    {real}* psi, const {real}* active, const unsigned long long* seeds,
    const {real}* gamma, const {real}* scattering, const {real}* strength,
    unsigned long long step, int spatial_n, int total_n, {real} dt_over_4v) {{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total_n) return;
    int member = j / spatial_n;
    unsigned long long seed = seeds[member];
    unsigned int a = (unsigned int)(j % spatial_n);
    unsigned int b = (unsigned int)step;
    unsigned int c = (unsigned int)(step >> 32);
    unsigned int d = 1U;
    philox4x32(&a, &b, &c, &d, (unsigned int)seed, (unsigned int)(seed >> 32));
    {uniforms}
    {real} radius = {sqrt_fn}(({real})(-2) * {log_fn}(u1));
    {real} angle = ({real})6.283185307179586476925286766559 * u2;
    {real} sn, cs;
    {sincos_fn}(angle, &sn, &cs);
    {real} sigma = {sqrt_fn}(strength[member] *
        (gamma[member] + scattering[member] * active[j]) * dt_over_4v);
    psi[2*j] += sigma * radius * cs;
    psi[2*j+1] += sigma * radius * sn;
}}
"""


def cuda_source(precision: str):
    """Render the CUDA source for one precision."""
    if precision == "single":
        uniforms = "float u1 = (float)((a >> 8) + 1U) * 0x1p-24f; float u2 = (float)((b >> 8) + 1U) * 0x1p-24f;"
        real, suffix, math_suffix = "float", "single", "f"
    elif precision == "double":
        uniforms = ("double u1 = ((double)(a >> 5) * 67108864.0 + (double)(b >> 6) + 1.0) * 0x1p-53; "
                    "double u2 = ((double)(c >> 5) * 67108864.0 + (double)(d >> 6) + 1.0) * 0x1p-53;")
        real, suffix, math_suffix = "double", "double", ""
    else:
        raise ValueError(f"Unknown precision {precision!r}")
    source = CUDA_SOURCE
    replacements = {"real": real, "suffix": suffix, "uniforms": uniforms,
                    "sqrt_fn": "sqrt" + math_suffix,
                    "log_fn": "log" + math_suffix,
                    "sincos_fn": "sincos" + math_suffix}
    for name, value in replacements.items():
        source = source.replace("{" + name + "}", value)
    return source.replace("{{", "{").replace("}}", "}")


WIGNER_DEVICE = r"""
__device__ __forceinline__ void wigner_add(
    {real}* re, {real}* im, unsigned long long seed, unsigned int cell,
    unsigned long long step, {real} strength, {real} gamma,
    {real} scattering, {real} active, {real} dt_over_4v) {{
    unsigned int a = cell;
    unsigned int b = (unsigned int)step;
    unsigned int c = (unsigned int)(step >> 32);
    unsigned int d = 1U;
    philox4x32(&a, &b, &c, &d, (unsigned int)seed, (unsigned int)(seed >> 32));
    {uniforms}
    {real} radius = {sqrt_fn}(({real})(-2) * {log_fn}(u1));
    {real} angle = ({real})6.283185307179586476925286766559 * u2;
    {real} sn, cs;
    {sincos_fn}(angle, &sn, &cs);
    {real} sigma = {sqrt_fn}(strength * (gamma + scattering * active) * dt_over_4v);
    *re += sigma * radius * cs;
    *im += sigma * radius * sn;
}}
"""


def device_source(precision: str):
    """Render the Philox and Wigner device functions for inclusion in a fused kernel."""
    philox = cuda_source(precision).split('extern "C" __global__ void philox_words', 1)[0]
    if precision == "single":
        uniforms = "float u1 = (float)((a >> 8) + 1U) * 5.9604644775390625e-08f; float u2 = (float)((b >> 8) + 1U) * 5.9604644775390625e-08f;"
        real, math_suffix = "float", "f"
    else:
        uniforms = ("double u1 = ((double)(a >> 5) * 67108864.0 + (double)(b >> 6) + 1.0) * 1.1102230246251565404236316680908203125e-16; "
                    "double u2 = ((double)(c >> 5) * 67108864.0 + (double)(d >> 6) + 1.0) * 1.1102230246251565404236316680908203125e-16;")
        real, math_suffix = "double", ""
    source = WIGNER_DEVICE
    replacements = {"real": real, "uniforms": uniforms,
                    "sqrt_fn": "sqrt" + math_suffix, "log_fn": "log" + math_suffix,
                    "sincos_fn": "sincos" + math_suffix}
    for name, value in replacements.items():
        source = source.replace("{" + name + "}", value)
    return philox + source.replace("{{", "{").replace("}}", "}")


def gpu_philox4x32(counters, key: int):
    """Return CUDA Philox words for testing the integer contract."""
    import cupy as cp

    source = cuda_source("single")
    kernel = cp.RawKernel(source, "philox_words")
    words = cp.ascontiguousarray(counters, dtype=cp.uint32)
    output = cp.empty_like(words)
    n = words.size // 4
    kernel(((n + 255) // 256,), (256,), (words, output, np.uint64(key), n))
    return output
