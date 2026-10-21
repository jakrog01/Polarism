"""Shared device combiner for separable multi-laser pump fields."""
from __future__ import annotations

import numpy as np


_SOURCE = """
typedef {real} real;
extern "C" __global__ void pump_combine(
    const real* env, int count, const real* coeffs, int coeff_offset,
    real* out, int n) {{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    real value = coeffs[coeff_offset] * env[j];
    for (int k = 1; k < count; ++k)
        value += coeffs[coeff_offset + k] * env[k*n + j];
    out[j] = value;
}}
"""


_BATCH_SOURCE = """
typedef {real} real;
extern "C" __global__ void pump_combine_batched(
    const real* env, const int* offsets, const int* sizes,
    int count, const real* coeffs, int coeff_offset, real* out, int n) {{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    real value = coeffs[coeff_offset] * env[offsets[0] + j % sizes[0]];
    for (int k = 1; k < count; ++k)
        value += coeffs[coeff_offset + k] * env[offsets[k] + j % sizes[k]];
    out[j] = value;
}}
"""


_MEMBER_SOURCE = """
typedef {real} real;
extern "C" __global__ void pump_combine_members(
    const real* env, const int* offsets, const int* sizes,
    int count, const real* coeffs, int coeff_offset, real* out,
    int n, int spatial_n) {{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    int member = j / spatial_n;
    real value = coeffs[coeff_offset + member] * env[offsets[0] + j % sizes[0]];
    for (int k = 1; k < count; ++k)
        value += coeffs[coeff_offset + k*(n/spatial_n) + member]
                 * env[offsets[k] + j % sizes[k]];
    out[j] = value;
}}
"""


class CudaPumpCombiner:
    """Keep stacked envelopes, coefficient storage, and reusable output arrays."""

    def __init__(self, xp, dtype, n):
        self.xp = xp
        self.dtype = dtype
        self.n = n
        self._envelopes = None
        self._kernel = xp.RawKernel(
            _SOURCE.format(real="float" if dtype == xp.float32 else "double"),
            "pump_combine", options=("--std=c++11", "--fmad=false"),
        )
        self._kernel.compile()

    def prepare(self, terms, outputs, samples, shape=None):
        envelopes = tuple(term[0] for term in terms)
        batched = shape is not None and len(shape) == 3
        member_coefficients = batched and any(np.ndim(term[1](0.0)) != 0 for term in terms)
        if (self._envelopes is None or len(envelopes) != len(self._envelopes)
                or any(a is not b for a, b in zip(envelopes, self._envelopes))
                or getattr(self, "_batched", None) != batched):
            if batched:
                arrays = [self.xp.ascontiguousarray(
                    self.xp.asarray(e, dtype=self.dtype)).ravel() for e in envelopes]
                sizes = np.asarray([a.size for a in arrays], dtype=np.int32)
                spatial_n = shape[-2] * shape[-1]
                if any(size not in (spatial_n, self.n) for size in sizes):
                    raise ValueError("pump envelope must be shared 2-D or batched 3-D")
                offsets = np.zeros(len(arrays), dtype=np.int32)
                offsets[1:] = np.cumsum(sizes[:-1])
                self.env = self.xp.concatenate(arrays)
                self.offsets = self.xp.asarray(offsets)
                self.sizes = self.xp.asarray(sizes)
                if not hasattr(self, "_batch_kernel"):
                    self._batch_kernel = self.xp.RawKernel(
                        _BATCH_SOURCE.format(real="float" if self.dtype == self.xp.float32
                                             else "double"),
                        "pump_combine_batched", options=("--std=c++11", "--fmad=false"),
                    )
                    self._batch_kernel.compile()
            else:
                self.env = self.xp.stack([
                    self.xp.ascontiguousarray(self.xp.asarray(e, dtype=self.dtype)).ravel()
                    for e in envelopes
                ])
            self._envelopes = envelopes
            self._batched = batched
        if member_coefficients and not hasattr(self, "_member_kernel"):
            self._member_kernel = self.xp.RawKernel(
                _MEMBER_SOURCE.format(real="float" if self.dtype == self.xp.float32
                                      else "double"),
                "pump_combine_members", options=("--std=c++11", "--fmad=false"),
            )
            self._member_kernel.compile()
        self._member_coefficients = member_coefficients
        self._members = shape[0] if batched else 1
        self._spatial_n = shape[-2] * shape[-1] if batched else self.n
        if not hasattr(self, "out") or len(self.out) != outputs:
            self.out = tuple(self.xp.empty(self.n, self.dtype) for _ in range(outputs))
        size = samples * len(terms) * (self._members if member_coefficients else 1)
        if not hasattr(self, "coeffs") or self.coeffs.size < size:
            self.coeffs = self.xp.empty(size, self.dtype)

    def coefficients(self, terms, times, stream):
        dtype = np.float32 if self.dtype == self.xp.float32 else np.float64
        if self._member_coefficients:
            host = np.asarray([
                np.broadcast_to((value.get() if hasattr(value, "get") else value),
                                (self._members,))
                for t in times for _, coefficient in terms
                for value in (coefficient(t),)
            ], dtype=dtype).ravel()
        else:
            host = np.asarray([coefficient(t) for t in times for _, coefficient in terms],
                              dtype=dtype)
        self.coeffs[:host.size].set(host, stream=stream)
        return host

    def combine(self, sample, output):
        count = len(self._envelopes)
        launch = ((self.n + 255) // 256,)
        block = (256,)
        if self._member_coefficients:
            self._member_kernel(launch, block,
                                (self.env, self.offsets, self.sizes, np.int32(count),
                                 self.coeffs, np.int32(sample * count * self._members),
                                 self.out[output], np.int32(self.n),
                                 np.int32(self._spatial_n)))
        elif self._batched:
            self._batch_kernel(launch, block,
                               (self.env, self.offsets, self.sizes, np.int32(count),
                                self.coeffs, np.int32(sample * count),
                                self.out[output], np.int32(self.n)))
        else:
            self._kernel(launch, block,
                         (self.env, np.int32(count), self.coeffs,
                          np.int32(sample * count), self.out[output], np.int32(self.n)))
        return self.out[output]

    @property
    def signature(self):
        return (len(self._envelopes), tuple(id(e) for e in self._envelopes),
                self._batched, self._member_coefficients,
                int(self.env.data.ptr), int(self.coeffs.data.ptr),
                int(self.offsets.data.ptr) if self._batched else 0,
                int(self.sizes.data.ptr) if self._batched else 0,
                *(int(out.data.ptr) for out in self.out))
