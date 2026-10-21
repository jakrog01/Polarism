"""High-level simulation setup and execution."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import warnings

import h5py
import numpy as np

from tqdm import tqdm

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine, has_full_rate_fp64
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.config.validation import validate_config
from polarism.grid.create_grid import create_grid
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.laser.laser_factory import LaserFactory
from polarism.laser.pump_field import PumpField
from polarism.plugins import load_plugins
from polarism.potential.create_potential import create_potential
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.results.real_time_visualization import RealTimeVisualization
from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup
from polarism.results.result_node import ResultNode
from polarism.results.observable_registry import available_observables
from polarism.results.results_manager import ResultsManager
from polarism.results.visitors.result_visitor import ResultVisitor
from polarism.results.visitors.animation_visitor import AnimationFieldSpec, AnimationVisitor
from polarism.results.visitors.storage_visitor import StorageVisitor, create_run_directory
from polarism.results.visitors.visualization_visitor import VisualizationVisitor
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.create_solver import create_solver
from polarism.solver.solver_compatibility import check_solver_compatibility

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np

    from polarism.config.simulation_parameters import Config


def _compute_density(**ctx):
    """Compute the condensate density."""
    xp = compute_engine.xp
    return xp.abs(ctx["state"].psi) ** 2


def _compute_total_norm(**ctx):
    """Compute the total condensate norm."""
    xp = compute_engine.xp
    grid = ctx["grid"]
    return (xp.abs(ctx["state"].psi) ** 2).sum() * grid.dx * grid.dy


def _compute_psi(**ctx):
    """Return the current condensate field."""
    return ctx["state"].psi


def _compute_pump_field(**ctx):
    """Return the full pump field."""
    return ctx["P_total"]


_fp64_notice_shown = False


class SimulationController:
    """Set up and run a full simulation."""
    cfg: Config
    grid: SimulationGrid2D
    boundary_condition: BoundaryCondition
    potential: Union["np.ndarray", "cp.ndarray"]
    lasers: list
    reservoir: AbstractReservoir
    state: SimulationState
    solver: AbstractSolver
    visualizer: RealTimeVisualization | None
    next_viz_time: float
    results_manager: ResultsManager
    storage_visitor: StorageVisitor | None
    animation_visitor: AnimationVisitor | None

    def __init__(self, cfg: Config):
        """Set up the simulation from the config."""
        self.cfg = cfg
        compute_engine.configure(cfg.compute_engine)
        self.xp = compute_engine.xp

        global _fp64_notice_shown
        if (compute_engine.use_gpu and cfg.solver.precision == "double"
                and not _fp64_notice_shown):
            properties = self.xp.cuda.runtime.getDeviceProperties(
                self.xp.cuda.runtime.getDevice()
            )
            if not has_full_rate_fp64(properties["major"], properties["minor"]):
                import sys
                name = properties["name"]
                if isinstance(name, bytes):
                    name = name.decode()
                print(
                    f"{name}: double precision runs at a fraction of single "
                    "precision speed on this GPU class; "
                    'solver.precision="single" is typically several times faster.',
                    file=sys.stderr,
                )
                _fp64_notice_shown = True

        load_plugins()
        validate_config(cfg)
        check_solver_compatibility(cfg)

        self.grid = create_grid(cfg.grid)
        self.boundary_condition = BoundaryCondition(
            self.grid, cfg.boundary_condition, cfg.physics, cfg.solver.dt
        )
        self.potential = create_potential(cfg.potential, self.grid)
        self.lasers = LaserFactory.create_laser(cfg.laser, self.grid.X, self.grid.Y, cfg.solver.precision)
        self.pump_field = PumpField(self.lasers, self.grid, cfg.solver.precision)
        self.reservoir = create_reservoir(cfg.reservoir, cfg.physics, self.grid, cfg.solver.precision)
        self.state = SimulationState(
            self.grid,
            cfg.physics.init_eps,
            cfg.solver.precision,
            mode=getattr(cfg.physics, "init_mode", "legacy_positive_uniform"),
            k_cutoff_um=getattr(cfg.physics, "init_k_cutoff_um", None),
            seed=getattr(cfg.physics, "init_seed", None),
        )
        self.solver = create_solver(cfg, self.grid)
        self.max_laser_power = self._compute_max_laser_power()

        cap = self.boundary_condition.before_step_action()
        if self.xp.iscomplexobj(cap) and not self.xp.iscomplexobj(self.potential):
            self.potential = self.potential.astype(self.state.psi.dtype)
            cap = cap.astype(self.state.psi.dtype)
        self.potential = self.potential + cap

        if cfg.solver.precision == "single":
            target = (
                complex_dtype(self.xp, "single")
                if self.xp.iscomplexobj(self.potential)
                else real_dtype(self.xp, "single")
            )
            self.potential = self.potential.astype(target)
        self.visualizer = None
        self.next_viz_time = 0.0
        self.results_manager = ResultsManager()
        self.storage_visitor = None
        self.animation_visitor = None
        self._step = 0
        self._state_shape_explicit = False
        self._stop_requested = False
        self._resumed = False
        self.run_directory = None

        if cfg.result.real_time_view:
            self._init_visualizer()

        if cfg.result.save_results:
            self._init_storage()

        if self.storage_visitor is not None:
            self.run_directory = self.storage_visitor.output_dir
        elif (cfg.checkpoint.every_steps or cfg.checkpoint.path is not None
              or cfg.checkpoint.resume_from is not None):
            self.run_directory = create_run_directory(cfg.result.output_directory)

        if cfg.result.animate:
            self._init_animation()

        if cfg.checkpoint.resume_from is not None:
            self.load_checkpoint(cfg.checkpoint.resume_from)

    def _compute_max_laser_power(self) -> float:
        """Return the largest configured laser power."""
        max_power = 0.0
        for laser in self.lasers:
            if hasattr(laser, "Pmax"):
                max_power = max(max_power, laser.Pmax)
            elif hasattr(laser, "P0"):
                max_power = max(max_power, laser.P0)
        return max_power if max_power > 0 else self.cfg.laser.Pmax

    def _init_visualizer(self) -> None:
        """Set up live result plotting."""
        extent = [
            self.grid.X.min(),
            self.grid.X.max(),
            self.grid.Y.min(),
            self.grid.Y.max(),
        ]

        nodes = self._build_result_nodes()

        fields_2d = []
        scalars = []
        for node in nodes:
            if not node.expose:
                continue
            if node.cmap is not None:
                fields_2d.append(Results2D(node.name, cmap=node.cmap, clim=node.clim, label=node.label))
            else:
                scalars.append(ResultScalar(node.name, label=node.label))

        scalar_groups = []
        if self.cfg.laser.expose_results:
            scalar_groups.append(
                ResultScalarGroup(
                    "P_lasers", [f"L{i}" for i in range(len(self.lasers))]
                )
            )

        self.visualizer = RealTimeVisualization(
            fields_2d=fields_2d,
            scalars=scalars,
            scalar_groups=scalar_groups,
            tmax=self.cfg.solver.total_time,
            grid_extent=extent,
        )

        self.results_manager.nodes = nodes
        self.add_visitor(VisualizationVisitor(self.visualizer))

    def _init_storage(self) -> None:
        """Set up result storage."""
        if not (
            self.cfg.result.formats
            or self.cfg.result.save_hdf5
            or self.cfg.result.save_json
            or self.cfg.result.save_npy
        ):
            return

        if not self.results_manager.nodes:
            self.results_manager.nodes = self._build_result_nodes()

        self.storage_visitor = StorageVisitor(
            self.cfg, x=compute_engine.to_cpu(self.grid.X[0, :]), y=compute_engine.to_cpu(self.grid.Y[:, 0])
        )
        self.add_visitor(self.storage_visitor)

    def _init_animation(self) -> None:
        """Set up the online animation visitor from ResultParameters config."""
        if not self.results_manager.nodes:
            self.results_manager.nodes = self._build_result_nodes()

        rcfg = self.cfg.result
        old_names = {"|ψ|²": "density", "N(t)": "norm", "Pump": "pump"}
        wanted = None
        if rcfg.animation_fields:
            wanted = set()
            for name in rcfg.animation_fields:
                if name in old_names:
                    warnings.warn(
                        f"animation_fields name {name!r} is deprecated; use {old_names[name]!r}",
                        DeprecationWarning, stacklevel=2,
                    )
                wanted.add(old_names.get(name, name))

        panel_specs = [
            AnimationFieldSpec(
                source=node.name,
                label=node.label or node.name,
                cmap=node.cmap,
                transform=None,
                clim=node.clim,
            )
            for node in self.results_manager.nodes
            if node.cmap is not None and (wanted is None or node.name in wanted)
        ]
        if not panel_specs:
            return

        import os
        out_path = rcfg.animation_output or os.path.join(
            rcfg.output_directory, "animation.mp4"
        )

        self.animation_visitor = AnimationVisitor(
            output_path=out_path,
            panel_specs=panel_specs,
            fps=rcfg.animation_fps,
            target_seconds=rcfg.animation_target_seconds,
            backend=rcfg.animation_backend,
            encoder_name=rcfg.animation_encoder,
        )
        self.add_visitor(self.animation_visitor)

    def add_visitor(self, visitor: ResultVisitor) -> None:
        """Register a visitor and build result nodes if this is the first one."""
        if not self.results_manager.nodes:
            self.results_manager.nodes = self._build_result_nodes()
        self.results_manager.add_visitor(visitor)

    def _build_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes for this run."""
        nodes = [
            ResultNode(
                name="density",
                label="|ψ|²",
                compute_fn=_compute_density,
                reduce_dim_fn=lambda f: float(f.sum()),
                cmap="magma",
                scaling=None,
                clim=None,
                expose=True,
                save=True,
                cut=None,
                is_field=True,
            ),
            ResultNode(
                name="norm",
                label="N(t)",
                compute_fn=_compute_total_norm,
                reduce_dim_fn=lambda v: v,
                cmap=None,
                scaling=None,
                clim=None,
                expose=True,
                save=True,
                cut=None,
                is_field=False,
            ),
        ]
        if hasattr(self.reservoir, "make_result_nodes"):
            nodes.extend(self.reservoir.make_result_nodes())

        if self.cfg.result.save_psi:
            nodes.append(
                ResultNode(
                    name="psi", compute_fn=_compute_psi, reduce_dim_fn=lambda f: f,
                    cmap=None, scaling=None, clim=None, expose=False, save=True,
                    cut=None, is_field=True,
                )
            )

        if self.cfg.laser.expose_results:
            nodes.append(
                ResultNode(
                    name="pump",
                    label="Pump",
                    compute_fn=_compute_pump_field,
                    reduce_dim_fn=lambda f: (
                        float(f.max()) if hasattr(f, "max") else float(f)
                    ),
                    cmap="inferno",
                    scaling=None,
                    clim=(0, self.max_laser_power),
                    expose=True,
                    save=True,
                    cut=None,
                    is_field=True,
                    static=all(laser.time_independent and laser.delay <= 0 for laser in self.lasers),
                )
            )

        for name in self.cfg.result.observables:
            produced = available_observables[name](self)
            nodes.extend(produced if isinstance(produced, list) else [produced])
        return nodes

    def _checkpoint_path(self) -> Path:
        """Return the configured path, creating a run directory when needed."""
        if self.cfg.checkpoint.path is not None:
            return Path(self.cfg.checkpoint.path)
        if self.run_directory is None:
            self.run_directory = create_run_directory(self.cfg.result.output_directory)
        return self.run_directory / "checkpoint.h5"

    def set_initial_state(self, psi, reservoir_state=None, t: float = 0.0) -> None:
        """Set a real-space initial field and optional reservoir densities."""
        shape = tuple(psi.shape)
        spatial = (self.grid.ny, self.grid.nx)
        if shape[-2:] != spatial or len(shape) not in (2, 3):
            raise ValueError(f"psi grid shape must end in {spatial}, got {shape}")
        if len(shape) == 3 and shape[0] < 1:
            raise ValueError("psi batch shape must have at least one member")
        if not np.isfinite(t) or t < 0:
            raise ValueError("t must be finite and nonnegative")
        step = round(t / self.cfg.solver.dt)
        if not np.isclose(t, step * self.cfg.solver.dt, rtol=0, atol=1e-12):
            raise ValueError("t must lie on a solver step boundary")
        dtype = complex_dtype(self.xp, self.cfg.solver.precision)
        self.state.psi = self.xp.asarray(psi, dtype=dtype).copy()
        self.state.t = step * self.cfg.solver.dt
        self._step = step
        self._state_shape_explicit = True
        if reservoir_state is not None:
            fields = tuple(reservoir_state)
            expected = self.reservoir.get_state()
            if len(fields) != len(expected):
                raise ValueError("reservoir state field count mismatch")
            if any(tuple(field.shape) != shape for field in fields):
                raise ValueError("reservoir batch or grid shape mismatch")
            self._restore_reservoir(fields)
        else:
            reset = getattr(self.reservoir, "reset", None)
            if callable(reset):
                if len(shape) == 3:
                    reset(shape[0])
                else:
                    reset()
            else:
                zeros = tuple(self.xp.zeros(shape, dtype=field.dtype)
                              for field in self.reservoir.get_state())
                self.reservoir.set_state(zeros)

    def _restore_reservoir(self, fields, *, cast: bool = True) -> None:
        """Restore saved densities without changing valid bit patterns."""
        from polarism.reservoir.double_reservoir import DoubleReservoir
        from polarism.reservoir.quadratic_double_reservoir import QuadraticDoubleReservoir
        from polarism.reservoir.single_reservoir import SingleReservoir

        dtype = real_dtype(self.xp, self.cfg.solver.precision) if cast else None
        arrays = tuple(self.xp.asarray(field, dtype=dtype).copy() for field in fields)
        if type(self.reservoir) is SingleReservoir:
            self.reservoir.nR = arrays[0]
        elif type(self.reservoir) is DoubleReservoir:
            self.reservoir.nA, self.reservoir.nI = arrays
        elif type(self.reservoir) is QuadraticDoubleReservoir:
            self.reservoir.nR, self.reservoir.nI = arrays
        else:
            self.reservoir.set_state(arrays)

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Atomically save the current step boundary to an HDF5 checkpoint."""
        from polarism import __version__

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        step = round(self.state.t / self.cfg.solver.dt)
        if self.state.t != step * self.cfg.solver.dt:
            raise ValueError("checkpoint requires a solver step boundary")
        self._step = step
        psi = compute_engine.to_cpu(self.state.psi)
        psi_k = self.state.psi_k
        fields = tuple(compute_engine.to_cpu(value) for value in self.reservoir.get_state())
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb"):
                pass
            with h5py.File(temporary, "w") as h5:
                h5.attrs["checkpoint_version"] = 1
                h5.attrs["polarism_version"] = __version__
                h5.attrs["step"] = self._step
                h5.attrs["t"] = self.state.t
                h5.attrs["dt"] = self.cfg.solver.dt
                h5.attrs["reservoir_type"] = self.cfg.reservoir.reservoir_type
                h5.attrs["config_json"] = json.dumps(asdict(self.cfg))
                h5.create_dataset("psi", data=psi)
                if psi_k is not None:
                    h5.create_dataset("psi_k", data=compute_engine.to_cpu(psi_k))
                group = h5.create_group("reservoir")
                for index, value in enumerate(fields):
                    group.create_dataset(str(index), data=value)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_checkpoint(self, path: str | os.PathLike) -> None:
        """Validate and restore a checkpoint for the current configuration."""
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("checkpoint_version") != 1:
                raise ValueError("unsupported checkpoint version")
            if h5.attrs["reservoir_type"] != self.cfg.reservoir.reservoir_type:
                raise ValueError("reservoir type mismatch")
            if h5.attrs["dt"] != self.cfg.solver.dt:
                raise ValueError("checkpoint dt mismatch")
            psi = h5["psi"][:]
            shape = psi.shape
            if shape[-2:] != (self.grid.ny, self.grid.nx) or len(shape) not in (2, 3):
                raise ValueError("checkpoint grid shape mismatch")
            if self._state_shape_explicit and shape != self.state.psi.shape:
                raise ValueError("checkpoint batch shape mismatch")
            fields = tuple(h5["reservoir"][str(i)][:] for i in range(len(h5["reservoir"])))
            if len(fields) != len(self.reservoir.get_state()) or any(x.shape != shape for x in fields):
                raise ValueError("checkpoint reservoir batch shape mismatch")
            step = int(h5.attrs["step"])
            t = float(h5.attrs["t"])
            if step < 0 or t != step * self.cfg.solver.dt:
                raise ValueError("checkpoint time and step mismatch")
            source_precision = json.loads(h5.attrs["config_json"])["solver"]["precision"]
            precision_changed = source_precision != self.cfg.solver.precision
            psi_k = h5["psi_k"][:] if "psi_k" in h5 else None
            if psi_k is not None and psi_k.shape != shape:
                raise ValueError("checkpoint spectral batch shape mismatch")
            if precision_changed:
                warnings.warn("checkpoint precision differs; casting state", UserWarning, stacklevel=2)
        dtype = complex_dtype(self.xp, self.cfg.solver.precision) if precision_changed else None
        self.state.psi = self.xp.asarray(psi, dtype=dtype).copy()
        self.state.t = t
        self._step = step
        self._state_shape_explicit = True
        self._restore_reservoir(fields, cast=precision_changed)
        self._resumed = True
        if psi_k is not None:
            self.state.psi_k = self.xp.asarray(psi_k, dtype=dtype).copy()
        self.solver = create_solver(self.cfg, self.grid)

    def run(self) -> None:
        """Run the simulation loop."""
        self.pump_field.lasers = self.lasers
        dt = self.cfg.solver.dt
        n_steps = round(self.cfg.solver.total_time / dt)
        if self._step > n_steps:
            raise ValueError("checkpoint step exceeds configured total_time")
        interval = self.cfg.checkpoint.every_steps
        checkpoint_enabled = (interval > 0 or self.cfg.checkpoint.path is not None
                              or self.cfg.checkpoint.resume_from is not None)
        last_checkpoint_step = -1
        previous_handlers = {}
        self._stop_requested = False

        def request_stop(signum, frame):
            self._stop_requested = True

        if self.cfg.checkpoint.on_signal and threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGUSR1):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, request_stop)

        try:
            if self._step == 0 and not self._resumed and (self.results_manager.visitors or self.storage_visitor is not None or self.animation_visitor is not None):
                self._record_results(self.state.t)
                if self.visualizer is not None:
                    self.next_viz_time += self.cfg.result.real_time_refresh_interval

            save_interval = max(1, self.cfg.result.save_interval)
            step = self._step
            with tqdm(total=n_steps, initial=step, desc="Simulating") as progress:
                while step < n_steps and not self._stop_requested:
                    recording = (
                        self.storage_visitor is not None
                        or self.animation_visitor is not None
                        or any(
                            not isinstance(visitor, VisualizationVisitor)
                            for visitor in self.results_manager.visitors
                        )
                    )
                    if self.visualizer is not None:
                        end = step + 1
                    elif recording:
                        end = min(n_steps, ((step // save_interval) + 1) * save_interval)
                    else:
                        end = n_steps
                    if interval:
                        end = min(end, ((step // interval) + 1) * interval)
                    if previous_handlers:
                        end = min(end, step + 64)
                    self.solver.advance(
                        self.potential, self.pump_field, self.reservoir,
                        self.boundary_condition, self.state, step, end - step,
                    )
                    progress.update(end - step)
                    step = end
                    self._step = step
                    t_after = step * dt
                    should_save = recording and step % save_interval == 0
                    should_viz = (
                        self.visualizer is not None and t_after >= self.next_viz_time
                    )

                    if self.visualizer is not None and self.visualizer._closed:
                        print(
                            f"\nVisualization closed at t={t_after:.3f}, "
                            "stopping simulation."
                        )
                        break

                    should_animate = (
                        self.animation_visitor is not None
                        and step % save_interval == 0
                    )

                    if should_save or should_viz or should_animate:
                        self._record_results(t_after)
                        if should_viz:
                            self.next_viz_time += self.cfg.result.real_time_refresh_interval
                    if interval and step % interval == 0:
                        self.save_checkpoint(self._checkpoint_path())
                        last_checkpoint_step = step
            if (checkpoint_enabled or self._stop_requested) and last_checkpoint_step != self._step:
                self.save_checkpoint(self._checkpoint_path())
        finally:
            try:
                for visitor in self.results_manager.visitors:
                    finalize = getattr(visitor, "finalize", None)
                    if not callable(finalize):
                        continue
                    if visitor is self.animation_visitor:
                        try:
                            finalize()
                        except Exception as exc:
                            import sys
                            print(f"WARNING: animation finalize failed: {exc}", file=sys.stderr)
                        continue
                    finalize()
                if self.storage_visitor is not None and self.storage_visitor not in self.results_manager.visitors:
                    self.storage_visitor.finalize()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)

    def _record_results(self, t: float) -> None:
        """Evaluate the pump and send the current state to result visitors."""
        P_total = self._compute_total_pump(t)
        self.results_manager.step(
            t,
            state=self.state,
            grid=self.grid,
            P_total=P_total,
            scalar_groups=self._get_scalar_groups(),
        )

    def _compute_total_pump(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        """Sum laser fields at time t; callers must treat the result as read-only."""
        self.pump_field.lasers = self.lasers
        return self.pump_field(t)

    def _get_scalar_groups(self) -> dict[str, dict[str, float]]:
        """Build scalar groups from the last computed laser fields.

        Must be called after _compute_total_pump so that each laser's .P
        is already set to the current time step.
        """
        if not self.cfg.laser.expose_results:
            return {}
        return {
            "P_lasers": {
                f"L{i}": self.xp.max(laser.P)
                for i, laser in enumerate(self.lasers)
            }
        }
