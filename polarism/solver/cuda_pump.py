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

    def prepare(self, terms, outputs, samples):
        envelopes = tuple(term[0] for term in terms)
        if (self._envelopes is None or len(envelopes) != len(self._envelopes)
                or any(a is not b for a, b in zip(envelopes, self._envelopes))):
            self.env = self.xp.stack([
                self.xp.ascontiguousarray(self.xp.asarray(e, dtype=self.dtype)).ravel()
                for e in envelopes
            ])
            self._envelopes = envelopes
        if not hasattr(self, "out") or len(self.out) != outputs:
            self.out = tuple(self.xp.empty(self.n, self.dtype) for _ in range(outputs))
        size = samples * len(terms)
        if not hasattr(self, "coeffs") or self.coeffs.size < size:
            self.coeffs = self.xp.empty(size, self.dtype)

    def coefficients(self, terms, times, stream):
        host = np.asarray([coefficient(t) for t in times for _, coefficient in terms],
                          dtype=np.float32 if self.dtype == self.xp.float32 else np.float64)
        self.coeffs[:host.size].set(host, stream=stream)
        return host

    def combine(self, sample, output):
        count = self.env.shape[0]
        self._kernel(((self.n + 255) // 256,), (256,),
                     (self.env, np.int32(count), self.coeffs,
                      np.int32(sample * count), self.out[output], np.int32(self.n)))
        return self.out[output]

    @property
    def signature(self):
        return (len(self._envelopes), tuple(id(e) for e in self._envelopes),
                int(self.env.data.ptr), int(self.coeffs.data.ptr),
                *(int(out.data.ptr) for out in self.out))
