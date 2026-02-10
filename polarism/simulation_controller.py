from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

from polarism.boundary_conditions.absorption.absorption_strategy import (
    AbsorptionStrategy,
)
from polarism.boundary_conditions.boundary_condition import BoundaryCondition
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
from polarism.simulation_grid_2D import SimulationGrid2D
from polarism.simulation_state import SimulationState
from polarism.solver.abstract_solver import AbstractSolver
from polarism.solver.create_solver import create_solver

if TYPE_CHECKING:
    from polarism.config.simulation_parameters import Config


def _compute_density(**ctx):
    return np.abs(ctx["state"].psi) ** 2


def _compute_total_norm(**ctx):
    return float((np.abs(ctx["state"].psi) ** 2).sum())


def _compute_pump_field(**ctx):
    return ctx["P_total"]


class SimulationController:
    cfg: Config
    grid: SimulationGrid2D
    boundary_condition: BoundaryCondition
    potential: AbsorptionStrategy
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

        self.grid = SimulationGrid2D(cfg.grid)
        self.boundary_condition = BoundaryCondition(
            self.grid, cfg.boundary_condition, cfg.physics
        )
        self.potential = create_potential(cfg.potential, self.grid)
        self.lasers = LaserFactory.create_laser(cfg.laser, self.grid.X, self.grid.Y)
        self.reservoir = create_reservoir(cfg.reservoir, cfg.physics, self.grid)
        self.state = SimulationState(self.grid)
        self.solver = create_solver(cfg, self.grid)

        self.potential += self.boundary_condition.before_step_action()

        self.visualizer = None
        self.next_viz_time = 0.0
        self.results_manager = ResultsManager()
        self.storage_visitor = None

        if cfg.result.real_time_view:
            self._init_visualizer()

        if cfg.result.save_results:
            self._init_storage()

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
                    name="P",
                    compute_fn=_compute_pump_field,
                    reduce_dim_fn=lambda f: (
                        float(f.sum()) if hasattr(f, "sum") else float(f)
                    ),
                    cmap="inferno",
                    scaling=None,
                    clim=None,
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
                self.state.t = t

                P_total = self._compute_total_pump(t)

                self.solver.step(
                    self.potential,
                    P_total,
                    self.reservoir,
                    self.boundary_condition,
                    self.state,
                )

                if self.storage_visitor:
                    self.results_manager.step(
                        t,
                        state=self.state,
                        P_total=P_total,
                        scalar_groups=(
                            self._get_scalar_groups(t)
                            if self.cfg.laser.expose_results
                            else {}
                        ),
                    )

                if self.visualizer and t >= self.next_viz_time:
                    self.results_manager.step(
                        t,
                        state=self.state,
                        P_total=P_total,
                        scalar_groups=(
                            self._get_scalar_groups(t)
                            if self.cfg.laser.expose_results
                            else {}
                        ),
                    )
                    self.next_viz_time += self.cfg.result.real_time_refresh_interval
        finally:
            if self.storage_visitor is not None:
                self.storage_visitor.finalize()

    def _compute_total_pump(self, t: float) -> np.ndarray:
        P_total = np.zeros_like(self.grid.X)
        for laser in self.lasers:
            P_total += laser.get_power(self.grid.X, self.grid.Y, t)
        return P_total

    def _get_scalar_groups(self, t: float) -> dict[str, dict[str, float]]:
        scalar_groups = {}
        if self.cfg.laser.expose_results:
            scalar_groups["P_lasers"] = {
                f"L{i}": float(np.sum(laser.get_power(self.grid.X, self.grid.Y, t)))
                for i, laser in enumerate(self.lasers)
            }
        return scalar_groups

    def _update_visualization(self, t: float, P_total: np.ndarray):
        scalar_groups = {}
        if self.cfg.laser.expose_results:
            scalar_groups["P_lasers"] = {
                f"L{i}": float(np.sum(laser.get_power(self.grid.X, self.grid.Y, t)))
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
