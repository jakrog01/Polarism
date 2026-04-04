from __future__ import annotations

from typing import TYPE_CHECKING, Union

from tqdm import trange

from polarism.boundary_conditions.boundary_condition import BoundaryCondition
from polarism.compute_engine import compute_engine
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
    xp = compute_engine.xp
    return xp.abs(ctx["state"].psi) ** 2


def _compute_total_norm(**ctx):
    xp = compute_engine.xp
    grid = ctx["grid"]
    return float((xp.abs(ctx["state"].psi) ** 2).sum()) * grid.dx * grid.dy


def _compute_pump_field(**ctx):
    return ctx["P_total"]


class SimulationController:
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

    def __init__(self, cfg: Config):
        self.cfg = cfg
        compute_engine.configure(cfg.compute_engine)
        self.xp = compute_engine.xp

        check_solver_compatibility(cfg)

        self.grid = create_grid(cfg.grid)
        self.boundary_condition = BoundaryCondition(
            self.grid, cfg.boundary_condition, cfg.physics
        )
        self.potential = create_potential(cfg.potential, self.grid)
        self.lasers = LaserFactory.create_laser(cfg.laser, self.grid.X, self.grid.Y)
        self.reservoir = create_reservoir(cfg.reservoir, cfg.physics, self.grid)
        self.state = SimulationState(self.grid, cfg.physics.init_eps)
        self.solver = create_solver(cfg, self.grid)
        self.max_laser_power = self._compute_max_laser_power()

        cap = self.boundary_condition.before_step_action()
        if self.xp.iscomplexobj(cap) and not self.xp.iscomplexobj(self.potential):
            self.potential = self.potential.astype(self.state.psi.dtype)
            cap = cap.astype(self.state.psi.dtype)
        self.potential = self.potential + cap
        self.visualizer = None
        self.next_viz_time = 0.0
        self.results_manager = ResultsManager()
        self.storage_visitor = None

        if cfg.result.real_time_view:
            self._init_visualizer()

        if cfg.result.save_results:
            self._init_storage()

    def _compute_max_laser_power(self) -> float:
        max_power = 0.0
        for laser in self.lasers:
            if hasattr(laser, "Pmax"):
                max_power = max(max_power, laser.Pmax)
            elif hasattr(laser, "P0"):
                max_power = max(max_power, laser.P0)
        return max_power if max_power > 0 else self.cfg.laser.Pmax

    def _init_visualizer(self) -> None:
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

    def _build_result_nodes(self) -> list[ResultNode]:
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

                if should_save or should_viz:
                    P_total = self._compute_total_pump(t_after)
                    self.results_manager.step(
                        t_after,
                        state=self.state,
                        grid=self.grid,
                        P_total=P_total,
                        scalar_groups=(
                            self._get_scalar_groups(t_after)
                            if self.cfg.laser.expose_results
                            else {}
                        ),
                    )
                    if should_viz:
                        self.next_viz_time += self.cfg.result.real_time_refresh_interval
        finally:
            if self.storage_visitor is not None:
                self.storage_visitor.finalize()

    def _compute_total_pump(self, t: float) -> Union[np.ndarray, cp.ndarray]:
        P_total = self.xp.zeros_like(self.grid.X)
        for laser in self.lasers:
            P_total += laser.get_power(self.grid.X, self.grid.Y, t)
        return P_total

    def _get_scalar_groups(self, t: float) -> dict[str, dict[str, float]]:
        scalar_groups = {}
        if self.cfg.laser.expose_results:
            scalar_groups["P_lasers"] = {
                f"L{i}": float(
                    self.xp.max(laser.get_power(self.grid.X, self.grid.Y, t))
                )
                for i, laser in enumerate(self.lasers)
            }
        return scalar_groups

    def _update_visualization(
        self, t: float, P_total: Union[np.ndarray, cp.ndarray]
    ) -> None:
        scalar_groups = {}
        if self.cfg.laser.expose_results:
            scalar_groups["P_lasers"] = {
                f"L{i}": float(
                    self.xp.max(laser.get_power(self.grid.X, self.grid.Y, t))
                )
                for i, laser in enumerate(self.lasers)
            }

        self.results_manager.step(
            t,
            grid=self.grid,
            state=self.state,
            lasers=self.lasers,
            reservoir=self.reservoir,
            scalar_groups=scalar_groups,
            P_total=P_total,
        )
