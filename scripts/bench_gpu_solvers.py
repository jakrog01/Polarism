"""Benchmark GPU solver steps with fixed traffic models and CUPTI activities."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import cupy as cp

from polarism.config.simulation_parameters import Config
from polarism.simulation_controller import SimulationController

SOLVERS = ("rk4-cuda", "ifrk4-fft-cuda", "split-step-fft-cuda")
RESERVOIRS = ("single", "double", "quadratic-double")
PRECISIONS = ("single", "double")
COLUMNS = ("solver", "grid", "reservoir", "precision", "ms/step (median)",
           "Mpts·step/s", "eff. GB/s", "% theor.", "% copy", "kernels/step",
           "memcpy/step", "memset/step", "syncs/step", "allocs/step", "GPU busy %")


def model_bytes_per_step(solver: str, grid: int, reservoir: str, precision: str) -> int:
    """Fixed reference traffic: cached stencil neighbors or FFT read/write pairs.

    This model deliberately stays constant across solver optimizations. FFT traffic
    is only a lower bound and omits pointwise operations and FFT scratch traffic.
    """
    fields = 1 if reservoir == "single" else 2
    real = 4 if precision == "single" else 8
    complex_size = 2 * real
    if solver == "rk4-cuda":
        rhs = (complex_size + fields * real + 3 * real) + (complex_size + fields * real)
        stage = (2 * complex_size + 2 * fields * real + 3 * real) + (complex_size + fields * real)
        combine = (5 * complex_size + 5 * fields * real) + (complex_size + fields * real)
        return grid * grid * (rhs + 3 * stage + combine)
    transforms = 2 if solver == "split-step-fft-cuda" else 9
    return grid * grid * (transforms * 2 * complex_size)


class AllocationHook(cp.cuda.MemoryHook):
    """Count requests to the CuPy pool, including reuse of cached blocks."""

    def __init__(self) -> None:
        self.count = 0

    def malloc_preprocess(self, **kwargs: Any) -> None:
        self.count += 1


def step(controller: SimulationController, profile: bool = False) -> None:
    if profile:
        cp.cuda.nvtx.RangePush("solver.step")
    try:
        pump = getattr(controller, "pump_field", controller._compute_total_pump)
        controller.solver.step(controller.potential, pump,
                               controller.reservoir, controller.boundary_condition,
                               controller.state)
        controller.state.t += controller.cfg.solver.dt
    finally:
        if profile:
            cp.cuda.nvtx.RangePop()


def synchronize() -> None:
    cp.cuda.runtime.deviceSynchronize()


def copy_bandwidth(grid: int, precision: str) -> float:
    """Measure a large device-to-device copy, counting read and write bytes."""
    nbytes = max(64 * 1024 * 1024, grid * grid * (8 if precision == "single" else 16))
    src = cp.empty(nbytes, dtype=cp.uint8)
    dst = cp.empty_like(src)
    src.fill(1)
    synchronize()
    samples = []
    for _ in range(3):
        synchronize()
        start = time.perf_counter()
        for _ in range(8):
            cp.copyto(dst, src)
        synchronize()
        samples.append(time.perf_counter() - start)
    del src, dst
    return 2 * nbytes * 8 / min(samples) / 1e9


def theoretical_bandwidth(props: dict, override: float | None) -> float | None:
    if override is not None:
        return override
    clock = props.get("memoryClockRate")
    width = props.get("memoryBusWidth")
    return clock * 1000 * width * 2 / 8 / 1e9 if clock and width else None


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment(props: dict) -> dict[str, Any]:
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    return {"gpu": name, "driver": cp.cuda.runtime.driverGetVersion(),
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__,
            "python": platform.python_version(), "date": datetime.now(timezone.utc).isoformat(),
            "git": git_metadata()}


class CuptiTrace:
    """Collect CUPTI activity records through its asynchronous buffer callbacks."""

    def __init__(self) -> None:
        from cupti import cupti
        self.api = cupti
        library = Path(cupti.__file__).parent.parent / "nvidia/cu13/lib/libcupti.so.13"
        self.lib = ctypes.CDLL(str(library))
        self.lib.cuptiActivityRegisterCallbacks.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self.lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
        self.lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self.lib.cuptiActivityGetNextRecord.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                        ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cuptiActivityGetNextRecord.restype = ctypes.c_int
        self.buffers: dict[int, Any] = {}
        self.records: list[dict[str, Any]] = []
        self.callback_error: Exception | None = None
        request_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        complete_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32,
                                         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t)
        self.request_callback = request_type(self._request)
        self.complete_callback = complete_type(self._complete)
        self.kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMCPY,
                      cupti.ActivityKind.MEMSET, cupti.ActivityKind.SYNCHRONIZATION)
        self._check(self.lib.cuptiActivityRegisterCallbacks(self.request_callback,
                                                             self.complete_callback))

    @staticmethod
    def _check(code: int) -> None:
        if code:
            raise RuntimeError(f"CUPTI error {code}")

    def _request(self, pointer: Any, size: Any, maximum: Any) -> None:
        buffer = ctypes.create_string_buffer(8 * 1024 * 1024 + 8)
        address = (ctypes.addressof(buffer) + 7) & ~7
        self.buffers[address] = buffer
        pointer[0] = address
        size[0] = 8 * 1024 * 1024
        maximum[0] = 0

    def _complete(self, context: int, stream: int, buffer: int, size: int, valid: int) -> None:
        try:
            record = ctypes.c_void_p()
            while valid and self.lib.cuptiActivityGetNextRecord(buffer, valid,
                                                                 ctypes.byref(record)) == 0:
                kind = ctypes.c_uint32.from_address(record.value).value
                api = self.api
                if kind == api.ActivityKind.CONCURRENT_KERNEL:
                    item = api.ActivityKernel13.from_ptr(record.value)
                    name = item.name
                    if isinstance(name, bytes):
                        name = name.decode(errors="replace")
                    self.records.append({"kind": "kernel", "name": str(name),
                                         "start": int(item.start), "end": int(item.end)})
                elif kind == api.ActivityKind.MEMCPY:
                    item = api.ActivityMemcpy7.from_ptr(record.value)
                    self.records.append({"kind": "memcpy", "type": int(item.copy_kind)})
                elif kind == api.ActivityKind.MEMSET:
                    self.records.append({"kind": "memset"})
                elif kind == api.ActivityKind.SYNCHRONIZATION:
                    item = api.ActivitySynchronization2.from_ptr(record.value)
                    self.records.append({"kind": "sync", "type": int(item.type)})
        except Exception as exc:
            self.callback_error = exc
        finally:
            self.buffers.pop(buffer, None)

    def start(self) -> None:
        self.records.clear()
        for kind in self.kinds:
            self._check(self.lib.cuptiActivityEnable(kind))

    def stop(self) -> None:
        self._check(self.lib.cuptiActivityFlushAll(0))
        for kind in self.kinds:
            self._check(self.lib.cuptiActivityDisable(kind))
        if self.callback_error is not None:
            raise RuntimeError("Failed to parse CUPTI activity records") from self.callback_error

    def summarize(self, steps: int, wall_seconds: float) -> dict[str, Any]:
        api = self.api
        kernels = [r for r in self.records if r["kind"] == "kernel"]
        counts = Counter(r["kind"] for r in self.records)
        copies = {name: sum(r["kind"] == "memcpy" and r["type"] == value
                            for r in self.records) / steps for name, value in
                  (("HtoD", api.ActivityMemcpyKind.HTOD),
                   ("DtoH", api.ActivityMemcpyKind.DTOH),
                   ("DtoD", api.ActivityMemcpyKind.DTOD))}
        syncs = {name: sum(r["kind"] == "sync" and r["type"] == value
                           for r in self.records) / steps for name, value in
                 (("stream", api.ActivitySynchronizationType.STREAM_SYNCHRONIZE),
                  ("event", api.ActivitySynchronizationType.EVENT_SYNCHRONIZE),
                  ("context", api.ActivitySynchronizationType.CONTEXT_SYNCHRONIZE),
                  ("stream-wait-event", api.ActivitySynchronizationType.STREAM_WAIT_EVENT))}
        by_name: dict[str, dict[str, float]] = defaultdict(lambda: {"count_per_step": 0.0,
                                                                      "total_us_per_step": 0.0})
        for record in kernels:
            stat = by_name[record["name"]]
            stat["count_per_step"] += 1 / steps
            stat["total_us_per_step"] += max(0, record["end"] - record["start"]) / steps / 1000
        intervals = sorted((r["start"], r["end"]) for r in kernels)
        merged_ns = 0
        if intervals:
            left, right = intervals[0]
            for start, end in intervals[1:]:
                if start > right:
                    merged_ns += right - left
                    left, right = start, end
                else:
                    right = max(right, end)
            merged_ns += right - left
        return {"kernels_per_step": counts["kernel"] / steps,
                "memcpy_per_step": counts["memcpy"] / steps, "memcpy_by_kind": copies,
                "memset_per_step": counts["memset"] / steps,
                "syncs_per_step": counts["sync"] / steps, "syncs_by_type": syncs,
                "gpu_busy_us_per_step": merged_ns / steps / 1000,
                "gpu_busy_pct": 100 * merged_ns / 1e9 / wall_seconds,
                "kernels_by_name": dict(sorted(by_name.items()))}


def trace_steps(controller: SimulationController, steps: int) -> dict[str, Any]:
    trace = CuptiTrace()
    synchronize()
    trace.start()
    start = time.perf_counter()
    for _ in range(steps):
        step(controller)
    synchronize()
    elapsed = time.perf_counter() - start
    trace.stop()
    return trace.summarize(steps, elapsed)


def benchmark_case(args: argparse.Namespace, solver: str, grid: int,
                   reservoir: str, precision: str, peak: float | None) -> dict[str, Any]:
    cfg = Config()
    cfg.compute_engine.use_gpu = True
    cfg.grid.nx = cfg.grid.ny = grid
    cfg.grid.grid_type = "periodic"
    cfg.solver.method = solver
    cfg.solver.precision = precision
    cfg.reservoir.reservoir_type = reservoir
    cfg.laser.laser_type = "pulse-gaussian"
    cfg.result.save_results = cfg.result.save_hdf5 = cfg.result.save_json = False
    cfg.result.save_npy = cfg.result.real_time_view = cfg.result.animate = False
    controller = SimulationController(cfg)
    for _ in range(args.warmup):
        step(controller)
    synchronize()
    steps = args.steps
    if steps is None:
        start = time.perf_counter()
        for _ in range(20):
            step(controller)
        synchronize()
        steps = max(20, math.ceil(0.5 / max((time.perf_counter() - start) / 20, 1e-9)))
    copy_gbps = copy_bandwidth(grid, precision)
    hook = AllocationHook()
    samples = []
    for _ in range(args.repeats):
        synchronize()
        with hook:
            if args.profile:
                cp.cuda.profiler.start()
            start = time.perf_counter()
            try:
                for _ in range(steps):
                    step(controller, args.profile)
                synchronize()
            finally:
                if args.profile:
                    cp.cuda.profiler.stop()
        samples.append((time.perf_counter() - start) / steps * 1000)
    median_ms = statistics.median(samples)
    traffic = model_bytes_per_step(solver, grid, reservoir, precision)
    effective = traffic / (median_ms / 1000) / 1e9
    record: dict[str, Any] = {"solver": solver, "grid": grid, "reservoir": reservoir,
                              "precision": precision, "steps": steps, "repeats": args.repeats,
                              "ms_per_step_samples": samples, "median_ms_per_step": median_ms,
                              "min_ms_per_step": min(samples),
                              "mpoints_steps_per_second": grid * grid / (median_ms / 1000) / 1e6,
                              "model_bytes_per_step": traffic,
                              "model_is_lower_bound": solver in {"ifrk4-fft-cuda", "split-step-fft-cuda"},
                              "effective_gbps": effective,
                              "theoretical_peak_gbps": peak,
                              "pct_theoretical": 100 * effective / peak if peak else None,
                              "copy_gbps": copy_gbps,
                              "pct_copy": 100 * effective / copy_gbps}
    record["allocs_per_step"] = hook.count / (steps * args.repeats)
    if not args.no_cupti:
        try:
            record.update(trace_steps(controller, args.cupti_steps))
        except ImportError:
            pass
    del controller
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return record


def fmt(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def markdown(cases: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for c in cases:
        cells = [c["solver"], str(c["grid"]), c["reservoir"], c["precision"],
                 fmt(c["median_ms_per_step"], 3), fmt(c["mpoints_steps_per_second"]),
                 fmt(c["effective_gbps"]) + ("*" if c["model_is_lower_bound"] else ""),
                 fmt(c["pct_theoretical"]), fmt(c["pct_copy"]),
                 fmt(c.get("kernels_per_step")), fmt(c.get("memcpy_per_step")),
                 fmt(c.get("memset_per_step")), fmt(c.get("syncs_per_step")),
                 fmt(c["allocs_per_step"]), fmt(c.get("gpu_busy_pct"))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\n*FFT effective bandwidth uses a transform traffic lower bound.\n"


def compare(before: Path, after: Path) -> str:
    old = json.loads(before.read_text(encoding="utf-8"))["cases"]
    new = json.loads(after.read_text(encoding="utf-8"))["cases"]
    key = lambda c: (c["solver"], c["grid"], c["reservoir"], c["precision"])
    old_by_key = {key(c): c for c in old}
    lines = ["| solver | grid | reservoir | precision | before ms/step | after ms/step | speedup | kernels before/after | syncs before/after |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in new:
        if key(c) not in old_by_key:
            continue
        b = old_by_key[key(c)]
        lines.append("| " + " | ".join((c["solver"], str(c["grid"]), c["reservoir"],
                     c["precision"], fmt(b["median_ms_per_step"], 3),
                     fmt(c["median_ms_per_step"], 3),
                     fmt(b["median_ms_per_step"] / c["median_ms_per_step"]),
                     f"{fmt(b.get('kernels_per_step'))} / {fmt(c.get('kernels_per_step'))}",
                     f"{fmt(b.get('syncs_per_step'))} / {fmt(c.get('syncs_per_step'))}")) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=SOLVERS)
    parser.add_argument("--grids", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--reservoirs", nargs="+", choices=RESERVOIRS, default=RESERVOIRS)
    parser.add_argument("--precisions", nargs="+", choices=PRECISIONS, default=PRECISIONS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--cupti-steps", type=int, default=10)
    parser.add_argument("--no-cupti", action="store_true")
    parser.add_argument("--peak-gbps", type=float)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/bench"))
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE.json", "AFTER.json"))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.compare:
        table = compare(*args.compare)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        output = args.out_dir / f"{args.compare[0].stem}_vs_{args.compare[1].stem}.md"
        output.write_text(table, encoding="utf-8")
        print(table, end="")
        print(f"Wrote {output}")
        return
    if any(n <= 0 for n in args.grids) or any(n <= 0 for n in
            (args.steps or 1, args.repeats, args.cupti_steps)) or args.warmup < 0:
        parser.error("grids, steps, repeats, and cupti-steps must be positive; warmup must be nonnegative")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    peak = theoretical_bandwidth(props, args.peak_gbps)
    if peak is None:
        print("Theoretical peak bandwidth: unknown (use --peak-gbps to supply it)")
    cases = []
    failures = []
    for solver in args.solvers:
        for grid in args.grids:
            for reservoir in args.reservoirs:
                for precision in args.precisions:
                    try:
                        case = benchmark_case(args, solver, grid, reservoir, precision, peak)
                    except Exception as exc:
                        failures.append({"solver": solver, "grid": grid, "reservoir": reservoir,
                                         "precision": precision, "error": f"{type(exc).__name__}: {exc}"})
                        print(f"{solver} {grid} {reservoir} {precision}: FAILED {failures[-1]['error']}", flush=True)
                        cp.get_default_memory_pool().free_all_blocks()
                        continue
                    cases.append(case)
                    print(f"{solver} {grid} {reservoir} {precision}: {case['median_ms_per_step']:.3f} ms/step", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = {"environment": environment(props), "cases": cases, "failures": failures}
    json_path = args.out_dir / f"{args.label}.json"
    markdown_path = args.out_dir / f"{args.label}.md"
    json_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(cases), encoding="utf-8")
    print(f"Wrote {json_path} and {markdown_path}")
    print(markdown(cases), end="")


if __name__ == "__main__":
    main()
