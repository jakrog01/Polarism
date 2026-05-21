"""High-level simulation setup and execution."""
from __future__ import annotations

from typing import TYPE_CHECKING, Union

from tqdm import trange

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
from polarism.config.dtype_utils import complex_dtype, real_dtype
from polarism.config.validation import validate_config
from polarism.grid.create_grid import create_grid
from polarism.grid.simulation_grid_2d import SimulationGrid2D
from polarism.laser.laser_factory import LaserFactory
from polarism.potential.create_potential import create_potential
from polarism.reservoir.abstract_reservoir import AbstractReservoir
from polarism.reservoir.create_reservoir import create_reservoir
from polarism.results.real_time_visualization import RealTimeVisualization
from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup
from polarism.results.result_node import ResultNode
from polarism.results.results_manager import ResultsManager
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
    return float((xp.abs(ctx["state"].psi) ** 2).sum()) * grid.dx * grid.dy


def _compute_pump_field(**ctx):
    """Return the full pump field."""
    return ctx["P_total"]


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

        validate_config(cfg)
        check_solver_compatibility(cfg)

        self.grid = create_grid(cfg.grid)
        self.boundary_condition = BoundaryCondition(
            self.grid, cfg.boundary_condition, cfg.physics
        )
        self.potential = create_potential(cfg.potential, self.grid)
        self.lasers = LaserFactory.create_laser(cfg.laser, self.grid.X, self.grid.Y, cfg.solver.precision)
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
                fields_2d.append(Results2D(node.name, cmap=node.cmap, clim=node.clim))
            else:
                scalars.append(ResultScalar(node.name))

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
        self.results_manager.add_visitor(VisualizationVisitor(self.visualizer))

    def _init_storage(self) -> None:
        """Set up result storage."""
        if not (
            self.cfg.result.save_hdf5
            or self.cfg.result.save_json
            or self.cfg.result.save_npy
        ):
            return

        if not self.results_manager.nodes:
            self.results_manager.nodes = self._build_result_nodes()

        self.storage_visitor = StorageVisitor(self.cfg)
        self.results_manager.add_visitor(self.storage_visitor)

    def _init_animation(self) -> None:
        """Set up the online animation visitor from ResultParameters config."""
        if not self.results_manager.nodes:
            self.results_manager.nodes = self._build_result_nodes()

        rcfg = self.cfg.result
        wanted = set(rcfg.animation_fields) if rcfg.animation_fields else None

        panel_specs = [
            AnimationFieldSpec(
                source=node.name,
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
        self.results_manager.add_visitor(self.animation_visitor)

    def _build_result_nodes(self) -> list[ResultNode]:
        """Build the result nodes for this run."""
        nodes = [
            ResultNode(
                name="|ψ|²",
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
                name="N(t)",
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

        if self.cfg.laser.expose_results:
            nodes.append(
                ResultNode(
                    name="Pump",
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
                )
            )

        return nodes

    def run(self) -> None:
        """Run the simulation loop."""
        dt = self.cfg.solver.dt
        n_steps = int(self.cfg.solver.total_time / dt)

        try:
            for step in trange(n_steps, desc="Simulating"):
                t = step * dt

                P_total = self._compute_total_pump(t)

                self.solver.step(
                    self.potential,
                    P_total,
                    self.reservoir,
                    self.boundary_condition,
                    self.state,
                )

                t_after = (step + 1) * dt
                self.state.t = t_after

                save_interval = max(1, self.cfg.result.save_interval)
                should_save = (
                    self.storage_visitor is not None
                    and step % save_interval == 0
                )
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
                    P_total = self._compute_total_pump(t_after)
                    self.results_manager.step(
                        t_after,
                        state=self.state,
                        grid=self.grid,
                        P_total=P_total,
                        scalar_groups=self._get_scalar_groups(),
                    )
                    if should_viz:
                        self.next_viz_time += self.cfg.result.real_time_refresh_interval
        finally:
            if self.storage_visitor is not None:
                self.storage_visitor.finalize()
            if self.animation_visitor is not None:
                try:
                    self.animation_visitor.finalize()
                except Exception as exc:
                    import sys
                    print(f"WARNING: animation finalize failed: {exc}", file=sys.stderr)

    def _compute_total_pump(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        """Sum the pump from all lasers at time t."""
        P_total = self.xp.zeros(self.grid.X.shape, dtype=real_dtype(self.xp, self.cfg.solver.precision))
        for laser in self.lasers:
            P_total += laser.get_power(self.grid.X, self.grid.Y, t)
        return P_total

    def _get_scalar_groups(self) -> dict[str, dict[str, float]]:
        """Build scalar groups from the last computed laser fields.

        Must be called after _compute_total_pump so that each laser's .P
        is already set to the current time step.
        """
        if not self.cfg.laser.expose_results:
            return {}
        return {
            "P_lasers": {
                f"L{i}": float(self.xp.max(laser.P))
                for i, laser in enumerate(self.lasers)
            }
        }
