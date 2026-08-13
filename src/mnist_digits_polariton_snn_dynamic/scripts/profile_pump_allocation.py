from __future__ import annotations

from dataclasses import dataclass
import gc
import statistics


@dataclass(frozen=True, slots=True)
class ProfileCase:
    name: str
    shape: tuple[int, ...]
    n_iter: int


def _event_time_ms(cp, fn) -> tuple[float, object]:
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    result = fn()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop)), result


def main() -> int:
    try:
        import cupy as cp
    except Exception as exc:
        print(f"CUPY_UNAVAILABLE {type(exc).__name__}: {exc}")
        return 0
    try:
        cp.cuda.Device(0).use()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        print(
            f"gpu={properties['name'].decode()} "
            f"free_gb={free_bytes / 1e9:.3f} total_gb={total_bytes / 1e9:.3f}"
        )
    except Exception as exc:
        print(f"GPU_UNAVAILABLE {type(exc).__name__}: {exc}")
        return 0
    cases = (
        ProfileCase("runner_single", (1024, 1024), 2000),
        ProfileCase("runner_batched_32", (32, 1024, 1024), 200),
    )
    print(
        "profile,shape,n_iter,alloc_ms_per_iter,inplace_ms_per_iter,"
        "delta_ms_per_iter,delta_percent,pool_used_before_mb,"
        "pool_used_after_alloc_mb,pool_used_after_inplace_mb"
    )
    for case in cases:
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        spatial = cp.ones(case.shape, dtype=cp.float64)
        factors = cp.linspace(0.1, 1.0, case.n_iter, dtype=cp.float64)
        buffer = cp.empty_like(spatial)
        cp.cuda.Stream.null.synchronize()
        before_mb = pool.used_bytes() / 1e6

        def alloc_loop():
            pump_t = None
            for index in range(case.n_iter):
                pump_t = factors[index] * spatial
            return pump_t

        def inplace_loop():
            for index in range(case.n_iter):
                cp.multiply(factors[index], spatial, out=buffer)
            return buffer

        alloc_samples: list[float] = []
        inplace_samples: list[float] = []
        after_alloc_mb = before_mb
        after_inplace_mb = before_mb
        for _ in range(3):
            elapsed_ms, pump_t = _event_time_ms(cp, alloc_loop)
            alloc_samples.append(elapsed_ms / case.n_iter)
            cp.cuda.Stream.null.synchronize()
            after_alloc_mb = pool.used_bytes() / 1e6
            del pump_t
            gc.collect()
            elapsed_ms, pump_t = _event_time_ms(cp, inplace_loop)
            inplace_samples.append(elapsed_ms / case.n_iter)
            cp.cuda.Stream.null.synchronize()
            after_inplace_mb = pool.used_bytes() / 1e6
        alloc_ms = float(statistics.median(alloc_samples))
        inplace_ms = float(statistics.median(inplace_samples))
        delta_ms = alloc_ms - inplace_ms
        delta_percent = 100.0 * delta_ms / alloc_ms if alloc_ms else 0.0
        print(
            f"{case.name},{'x'.join(str(value) for value in case.shape)},"
            f"{case.n_iter},{alloc_ms:.6f},{inplace_ms:.6f},"
            f"{delta_ms:.6f},{delta_percent:.2f},{before_mb:.1f},"
            f"{after_alloc_mb:.1f},{after_inplace_mb:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
