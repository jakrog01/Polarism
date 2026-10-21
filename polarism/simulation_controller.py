"""High-level simulation setup and execution."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union
import warnings

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
from polarism.results.visitors.storage_visitor import StorageVisitor
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

        if cfg.result.real_time_view:
            self._init_visualizer()

        if cfg.result.save_results:
            self._init_storage()

        if cfg.result.animate:
            self._init_animation()

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

    def run(self) -> None:
        """Run the simulation loop."""
        self.pump_field.lasers = self.lasers
        dt = self.cfg.solver.dt
        n_steps = round(self.cfg.solver.total_time / dt)

        try:
            if self.results_manager.visitors or self.storage_visitor is not None or self.animation_visitor is not None:
                self._record_results(self.state.t)
                if self.visualizer is not None:
                    self.next_viz_time += self.cfg.result.real_time_refresh_interval

            save_interval = max(1, self.cfg.result.save_interval)
            step = 0
            with tqdm(total=n_steps, desc="Simulating") as progress:
                while step < n_steps:
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
                    self.solver.advance(
                        self.potential, self.pump_field, self.reservoir,
                        self.boundary_condition, self.state, step, end - step,
                    )
                    progress.update(end - step)
                    step = end
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
        finally:
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
