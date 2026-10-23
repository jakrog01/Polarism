"""Parameter sweep execution and batch-size selection."""
from __future__ import annotations

from itertools import product
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time

from polarism.compute_engine import compute_engine
from polarism.plugins import load_plugins
from polarism.simulation_controller import SimulationController
from polarism.results.reader import load_results
from polarism.results.visitors.storage_visitor import create_run_directory
from polarism.sweep_config import apply_overrides, override_kind


def autotune_batch_size(limit, timer, *, max_steps=16384, initial_steps=8):
    """Choose the measured maximum throughput, stopping after two drops.

    ``timer(batch_size, steps)`` warms and times that number of steps.
    """
    choices = [1]
    while choices[-1] * 2 <= limit:
        choices.append(choices[-1] * 2)
    if choices[-1] != limit:
        choices.append(limit)
    best, best_rate, drops, previous = 1, 0.0, 0, 0.0
    for size in choices:
        steps = min(initial_steps, max_steps)
        while True:
            elapsed = timer(size, steps)
            if elapsed >= 0.05 or steps >= max_steps:
                break
            steps = min(steps * 2, max_steps)
        rate = size * steps / max(elapsed, 1e-12)
        if rate > best_rate:
            best, best_rate = size, rate
        drops = drops + 1 if rate < previous else 0
        if drops >= 2:
            break
        previous = rate
    return best


def _cache_file():
    root = os.environ.get("XDG_CACHE_HOME")
    return (Path(root) if root else Path.home() / ".cache") / "polarism" / "sweep_batches.json"


def _write_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".sweep_batches_", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(cache, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _array_bytes(obj):
    seen = set()
    total = 0
    for value in vars(obj).values():
        if (not isinstance(value, type) and hasattr(value, "nbytes")
                and hasattr(value, "shape") and id(value) not in seen):
            total += value.nbytes
            seen.add(id(value))
    return total


def _json_value(value):
    """Convert NumPy scalars while preserving the manifest's numeric values."""
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Sweep value {value!r} cannot be encoded as JSON")


def _config_signature(cfg):
    """Capture settings that determine stored simulation frames."""
    data = asdict(cfg)
    return {name: data[name] for name in ("grid", "solver", "reservoir", "physics",
                                         "laser", "potential", "boundary_condition",
                                         "compute_engine")}


class Sweep:
    """Run Cartesian-product or zipped config overrides in deterministic order.

    Multiple-laser entries use ``lasers[1].delay`` or
    ``lasers[1].extra.name`` with zero-based indices into the laser YAML file.
    ``run`` returns saved sweep results when output is enabled, otherwise a
    list of dictionaries with final ``psi`` and ``reservoir`` arrays.
    """

    def __init__(self, cfg, parameters, mode="product"):
        load_plugins()
        if mode not in {"product", "zip"}:
            raise ValueError("mode must be 'product' or 'zip'")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must map dotted paths to sequences")
        keys = list(parameters)
        for key in keys:
            override_kind(key)
        values = [list(parameters[key]) for key in keys]
        if any(not values_for_key for values_for_key in values):
            raise ValueError("sweep parameter sequences must be nonempty")
        if mode == "zip" and len({len(value) for value in values}) > 1:
            raise ValueError("zip sweep parameter sequences must have equal lengths")
        rows = product(*values) if mode == "product" else zip(*values)
        self.cfg = cfg
        self.mode = mode
        self.members = [dict(zip(keys, row)) for row in rows]
        if cfg.result.view_member >= len(self.members):
            raise ValueError("result.view_member is outside the sweep")
        self.groups = []
        grouped = {}
        for index, member in enumerate(self.members):
            apply_overrides(cfg, member, allow_groups=True)
            group_values = {key: value for key, value in member.items()
                            if override_kind(key) == "group"}
            signature = json.dumps(group_values, sort_keys=True, default=str)
            if signature not in grouped:
                grouped[signature] = {"overrides": group_values, "indices": []}
                self.groups.append(grouped[signature])
            grouped[signature]["indices"].append(index)

    def _gpu_batch_size(self, cfg, overrides):
        import cupy as cp

        probe_cfg = deepcopy(cfg)
        probe_cfg.result.save_results = False
        probe_cfg.result.real_time_view = False
        probe_cfg.result.animate = False
        probe_cfg.result.view_member = 0
        probe_cfg.checkpoint.every_steps = 0
        probe_cfg.checkpoint.path = None
        probe_cfg.checkpoint.resume_from = None
        count = len(overrides)
        settings = cfg.sweep
        if settings.batch_size is not None:
            return min(count, settings.batch_size)
        free, _ = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        used_before = pool.used_bytes()
        probe = SimulationController(probe_cfg, members=[overrides[0]])
        measured = max(0, pool.used_bytes() - used_before)
        getter = getattr(probe.solver, "bytes_per_member", None)
        if callable(getter):
            bytes_per_member = getter()
        elif getter is not None:
            bytes_per_member = getter
        else:
            bytes_per_member = max(measured, sum(_array_bytes(item) for item in
                                   (probe.solver, probe.state, probe.reservoir)))
            bytes_per_member += probe.grid.nx * probe.grid.ny * (32 if cfg.solver.precision == "double" else 16)
        del probe
        limit = min(count, max(1, int(free * settings.memory_fraction // max(1, bytes_per_member))))
        if settings.max_batch is not None:
            limit = min(limit, settings.max_batch)
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"]
        if isinstance(device, bytes):
            device = device.decode()
        names = sorted(key for member in overrides for key in member
                       if key.startswith("physics.") and key not in
                       {"physics.init_seed", "physics.init_mode", "physics.init_k_cutoff_um", "physics.init_eps"})
        pump_kind = cfg.laser.laser_type if cfg.laser.mode == "single" else "multiple"
        cache_key = json.dumps([device, cfg.solver.method, [cfg.grid.ny, cfg.grid.nx],
                                cfg.solver.precision, cfg.reservoir.reservoir_type,
                                sorted(set(names)), pump_kind])
        cache_file = _cache_file()
        try:
            cache = json.loads(cache_file.read_text())
        except (OSError, ValueError):
            cache = {}
        cached = cache.get(cache_key)
        if (isinstance(cached, dict) and cached.get("version") == 2
                and cached.get("limit", 0) >= limit):
            return min(limit, cached["choice"])

        max_steps = min(16384, max(1, round(cfg.solver.total_time / cfg.solver.dt)))

        def timer(size, steps):
            controller = SimulationController(probe_cfg, members=overrides[:size])
            solver = controller.solver
            args = (controller.potential, controller.pump_field, controller.reservoir,
                    controller.boundary_condition, controller.state)
            solver.advance(*args, 0, steps)
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            solver.advance(*args, steps, steps)
            cp.cuda.Stream.null.synchronize()
            return max(time.perf_counter() - start, 1e-12)

        choice = autotune_batch_size(limit, timer, max_steps=max_steps)
        try:
            cache[cache_key] = {"choice": choice, "limit": limit, "version": 2}
            _write_cache(cache_file, cache)
        except OSError:
            pass
        return choice

    def run(self, resume: str | Path | None = None):
        """Execute batches, optionally resuming a sweep directory."""
        from polarism import __version__

        checkpoint = self.cfg.checkpoint
        source = resume or checkpoint.resume_from
        output = self.cfg.result.save_results
        if output and not (self.cfg.result.formats or self.cfg.result.save_hdf5
                           or self.cfg.result.save_npy or self.cfg.result.save_json):
            raise ValueError("sweep output requires at least one storage format")
        checkpoint_enabled = bool(checkpoint.every_steps or checkpoint.path or source)
        run_dir = Path(source) if source else (
            create_run_directory(self.cfg.result.output_directory)
            if output or checkpoint_enabled else None
        )
        self.run_directory = run_dir
        manifest_path = run_dir / "sweep.json" if run_dir is not None else None
        if source:
            manifest = json.loads(manifest_path.read_text())
            normalized = json.loads(json.dumps(self.members, default=_json_value))
            if manifest["parameters"] != normalized or manifest["mode"] != self.mode:
                raise ValueError("sweep parameters or mode differ from the saved sweep")
            if manifest.get("config_signature") != _config_signature(self.cfg):
                raise ValueError("sweep simulation config differs from the saved sweep")
            planned = manifest["batches"]
        else:
            planned = []
            for group_index, group in enumerate(self.groups):
                cfg = apply_overrides(self.cfg, group["overrides"], allow_groups=True)
                indices = group["indices"]
                overrides = [{key: value for key, value in self.members[index].items()
                              if override_kind(key) == "batch"} for index in indices]
                compute_engine.configure(cfg.compute_engine)
                size = (self._gpu_batch_size(cfg, overrides) if compute_engine.use_gpu
                        else cfg.sweep.batch_size or len(indices))
                if cfg.sweep.max_batch is not None:
                    size = min(size, cfg.sweep.max_batch)
                for offset in range(0, len(indices), size):
                    planned.append({"group": group_index, "indices": indices[offset:offset + size],
                                    "size": len(indices[offset:offset + size])})
            if manifest_path is not None:
                mapping = [None] * len(self.members)
                for batch_index, batch in enumerate(planned):
                    for position, index in enumerate(batch["indices"]):
                        mapping[index] = {"group": batch["group"], "batch": batch_index,
                                          "position": position}
                manifest = {"format_version": 1, "polarism_version": __version__,
                            "mode": self.mode, "parameter_keys": list(self.members[0]),
                            "parameters": self.members, "members": mapping,
                            "batches": planned, "batch_sizes": [batch["size"] for batch in planned],
                            "config_signature": _config_signature(self.cfg)}
                manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_value))

        results = [None] * len(self.members)
        for batch_index, batch in enumerate(planned):
            group = self.groups[batch["group"]]
            cfg = apply_overrides(self.cfg, group["overrides"], allow_groups=True)
            indices = batch["indices"]
            local = [{key: value for key, value in self.members[index].items()
                      if override_kind(key) == "batch"} for index in indices]
            if cfg.result.real_time_view or cfg.result.animate:
                selected = self.cfg.result.view_member
                cfg.result.real_time_view &= selected in indices
                cfg.result.animate &= selected in indices
                cfg.result.view_member = indices.index(selected) if selected in indices else 0
            else:
                cfg.result.view_member = 0
            batch_dir = run_dir / f"batch_{batch_index:03d}" if run_dir is not None else None
            if cfg.result.animate and run_dir is not None and cfg.result.animation_output is None:
                cfg.result.animation_output = str(run_dir / "animation.mp4")
            if batch_dir is not None:
                batch_dir.mkdir(parents=True, exist_ok=True)
                if output and (batch_dir / "complete").exists():
                    continue
            if checkpoint_enabled and batch_dir is not None:
                cfg.checkpoint.path = str(batch_dir / "checkpoint.h5")
                cfg.checkpoint.resume_from = (cfg.checkpoint.path if
                                               (batch_dir / "checkpoint.h5").exists() else None)
            else:
                cfg.checkpoint.resume_from = None
            controller = SimulationController(cfg, members=local, output_dir=batch_dir if output else None,
                                              member_indices=indices)
            controller.run()
            if controller._stop_requested:
                return None
            if batch_dir is not None:
                (batch_dir / "complete").touch()
            if not output:
                psi = controller.state.psi
                fields = tuple(field[:, 0] if controller.state.batched else field[0, 0]
                               for field in controller.reservoir.get_state())
                for position, index in enumerate(indices):
                    member_psi = psi[position] if len(local) > 1 else psi
                    member_fields = tuple(field[position] if len(local) > 1 else field
                                          for field in fields)
                    results[index] = {"psi": member_psi.copy(),
                                      "reservoir": tuple(field.copy() for field in member_fields)}
        return load_results(run_dir) if output else results
