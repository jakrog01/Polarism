"""Compare CUDA per-step calls with chunked solver execution."""
from __future__ import annotations

import argparse
import json
import sys
import time

from polarism.compute_engine import has_cuda_device


def _controller(method: str, size: int, precision: str, reservoir: str,
                tableau: str = "rk4", batch: int = 1):
    from polarism.config.simulation_parameters import Config
    from polarism.simulation_controller import SimulationController

    cfg = Config()
    cfg.grid.nx = cfg.grid.ny = size
    cfg.grid.lx = cfg.grid.ly = 100.0
    cfg.grid.grid_type = "periodic"
    cfg.reservoir.reservoir_type = reservoir
    cfg.potential.potential_type = "zero"
    cfg.laser.mode = "single"
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.P0 = 0.02
    cfg.laser.sigma_space = 12.0
    cfg.solver.method = method
    cfg.solver.tableau = tableau
    cfg.solver.dt = 1e-3
    cfg.solver.total_time = 1.0
    cfg.solver.precision = precision
    cfg.compute_engine.use_gpu = True
    controller = SimulationController(cfg, members=[{} for _ in range(batch)] if batch > 1 else None)
    controller.pump_field.lasers = controller.lasers
    return controller


def _arguments(controller):
    return (controller.potential, controller.pump_field, controller.reservoir,
            controller.boundary_condition, controller.state)


def _time_run(method, size, precision, reservoir, steps, chunked, cp, tableau="rk4"):
    controller = _controller(method, size, precision, reservoir, tableau)
    args = _arguments(controller)
    dt = controller.cfg.solver.dt
    if chunked:
        def chunks(first_step):
            done = 0
            while done < steps:
                length = min(64, steps - done)
                controller.solver.advance(*args, first_step + done, length)
                done += length

        # The untimed pass captures every graph length the timed pass uses.
        chunks(0)
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        chunks(steps)
    else:
        for k in range(4):
            controller.solver.step(*args)
            controller.state.t = (k + 1) * dt
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        for k in range(steps):
            controller.solver.step(*args)
            controller.state.t = (k + 5) * dt
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - start) * 1e6 / steps


def _time_batched(method, size, precision, reservoir, steps, batch, cp, tableau="rk4"):
    def timed_advance(controller):
        args = _arguments(controller)
        controller.solver.advance(*args, 0, steps)
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        controller.solver.advance(*args, steps, steps)
        cp.cuda.Device().synchronize()
        return time.perf_counter() - start

    batched = _controller(method, size, precision, reservoir, tableau, batch)
    batched_seconds = timed_advance(batched)
    del batched
    sequential_seconds = 0.0
    for _ in range(batch):
        single = _controller(method, size, precision, reservoir, tableau)
        sequential_seconds += timed_advance(single)
        del single
    return sequential_seconds, batched_seconds


def _copy_bandwidth(cp):
    source = cp.empty(256 * 1024 * 1024, dtype=cp.uint8)
    target = cp.empty_like(source)

    def copy():
        cp.cuda.runtime.memcpy(target.data.ptr, source.data.ptr, source.nbytes,
                               cp.cuda.runtime.memcpyDeviceToDevice)

    copy()
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    for _ in range(16):
        copy()
    cp.cuda.Device().synchronize()
    # Each copy reads and writes every byte.
    return 2 * source.nbytes * 16 / (time.perf_counter() - start) / 1e9


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["rk4-cuda", "split-step-fft-cuda"])
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--precisions", nargs="+", choices=("single", "double"),
                        default=["single", "double"])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--reservoir", choices=("single", "double", "quadratic-double"),
                        default="single")
    parser.add_argument("--tableau", choices=("rk4", "dp5", "dop853", "vern7", "vern9"),
                        default="rk4")
    parser.add_argument("--batch", type=int, help="compare one batch with this many sequential runs")
    args = parser.parse_args(argv)
    if args.steps < 1 or any(n < 2 for n in args.grid_sizes) or (args.batch is not None and args.batch < 2):
        parser.error("steps must be positive, grid sizes at least 2, and batch at least 2")
    if not has_cuda_device():
        print("polarism-bench requires CuPy and an available CUDA GPU.", file=sys.stderr)
        return 1

    import cupy as cp

    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    capability = f'{properties["major"]}.{properties["minor"]}'
    runtime = cp.cuda.runtime.runtimeGetVersion()
    driver = cp.cuda.runtime.driverGetVersion()
    bandwidth = _copy_bandwidth(cp)
    print(f"GPU: {name}; compute capability {capability}; CUDA runtime {runtime}; "
          f"driver {driver}; CuPy {cp.__version__}; copy {bandwidth:.1f} GB/s")
    for method in args.methods:
        for precision in args.precisions:
            for size in args.grid_sizes:
                if args.batch is not None:
                    try:
                        sequential_seconds, batched_seconds = _time_batched(
                            method, size, precision, args.reservoir, args.steps, args.batch, cp,
                            args.tableau)
                    except cp.cuda.memory.OutOfMemoryError:
                        cp.get_default_memory_pool().free_all_blocks()
                        print(f"{method:22} {precision:6} {size:5}  batch {args.batch:4}  "
                              "skipped: out of device memory")
                        continue
                    record = {
                        "method": method, "tableau": args.tableau, "precision": precision,
                        "grid": size, "reservoir": args.reservoir, "steps": args.steps,
                        "batch": args.batch, "sequential_s": sequential_seconds,
                        "batched_s": batched_seconds,
                        "speedup": sequential_seconds / batched_seconds,
                        "ns_per_member_element_step": batched_seconds * 1e9 /
                        (args.batch * size * size * args.steps),
                        "device": name, "compute_capability": capability,
                        "cuda_runtime": runtime, "cuda_driver": driver,
                        "cupy": cp.__version__, "copy_gbps": bandwidth,
                    }
                    print(f"{method:22} {precision:6} {size:5}  batch {args.batch:4}  "
                          f"{record['speedup']:5.2f}x  "
                          f"{record['ns_per_member_element_step']:.3f} ns/member·element·step")
                    print(json.dumps(record, sort_keys=True))
                    continue
                step_us = _time_run(method, size, precision, args.reservoir,
                                    args.steps, False, cp, args.tableau)
                advance_us = _time_run(method, size, precision, args.reservoir,
                                       args.steps, True, cp, args.tableau)
                record = {
                    "method": method, "tableau": args.tableau, "precision": precision, "grid": size,
                    "reservoir": args.reservoir, "steps": args.steps,
                    "step_us": step_us, "advance_us": advance_us,
                    "speedup": step_us / advance_us,
                    "mpts_steps_per_s": size * size / advance_us,
                    "device": name, "compute_capability": capability,
                    "cuda_runtime": runtime, "cuda_driver": driver,
                    "cupy": cp.__version__, "copy_gbps": bandwidth,
                }
                print(f"{method:22} {precision:6} {size:5}  "
                      f"step {step_us:9.1f} us  advance {advance_us:9.1f} us  "
                      f"{record['speedup']:5.2f}x  {record['mpts_steps_per_s']:8.1f} Mpt·step/s")
                print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
