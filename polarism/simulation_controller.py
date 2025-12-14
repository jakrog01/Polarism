import numpy as np

import polarism as ps
from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup


class SimulationController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.grid = ps.SimulationGrid2D(cfg.grid)
        self.boundry_condition = ps.BoundaryCondition(
            self.grid, cfg.boundry_condition, cfg.physics
        )
        self.potential = ps.create_potential(cfg.potential, self.grid)
        self.lasers = ps.LaserFactory.create_laser(cfg.laser)
        self.reservoir = ps.create_reservoir(cfg.reservoir, cfg.physics, self.grid)
        self.state = ps.SimulationState(self.grid)
        self.solver = ps.create_solver(cfg, self.grid)

        self.potential += self.boundry_condition.before_step_action()

        self.visualizer = None
        self.next_viz_time = 0.0

        if cfg.result.real_time_view:
            self._init_visualizer()

    def _init_visualizer(self):
        extent = [
            self.grid.X.min(),
            self.grid.X.max(),
            self.grid.Y.min(),
            self.grid.Y.max(),
        ]

        fields_2d = [Results2D("|psi|^2", cmap="magma")]
        scalars = [ResultScalar("N(t)")]

        if self.cfg.reservoir.expose_results:
            if self.cfg.reservoir.reservoir_type == "single":
                fields_2d.append(Results2D("nR", cmap="viridis"))
                scalars.append(ResultScalar("nR_max"))
            else:
                fields_2d.extend(
                    [
                        Results2D("nA", cmap="viridis"),
                        Results2D("nI", cmap="plasma"),
                    ]
                )
                scalars.extend(
                    [
                        ResultScalar("nA_max"),
                        ResultScalar("nI_max"),
                    ]
                )

        if self.cfg.laser.expose_results:
            fields_2d.append(Results2D("P", cmap="inferno"))
            scalar_groups = [
                ResultScalarGroup(
                    "P_lasers", [f"L{i}" for i in range(len(self.lasers))]
                )
            ]
        else:
            scalar_groups = []

        self.visualizer = ps.RealTimeVisualization(
            fields_2d=fields_2d,
            scalars=scalars,
            scalar_groups=scalar_groups,
            tmax=self.cfg.solver.total_time,
            grid_extent=extent,
        )

    def run(self):
        t = 0.0
        while t < self.cfg.solver.total_time:
            P_total = np.zeros_like(self.grid.X)
            for laser in self.lasers:
                P_total += laser.get_power(self.grid.X, self.grid.Y, t)

            self.solver.step(
                self.potential,
                P_total,
                self.reservoir,
                self.boundry_condition,
                self.state,
            )

            if self.visualizer and t >= self.next_viz_time:
                density = np.abs(self.state.psi) ** 2

                fields = {
                    "|psi|^2": density,
                    "P": P_total,
                }

                scalars = {
                    "N(t)": density.sum(),
                }

                if self.cfg.reservoir.expose_results:
                    if self.cfg.reservoir.reservoir_type == "single":
                        nR = self.reservoir.get_reservoir_density()
                        fields["nR"] = nR
                        scalars["nR_max"] = nR.max()
                    else:
                        nA, nI = self.reservoir.get_reservoir_densities()
                        fields["nA"] = nA
                        fields["nI"] = nI
                        scalars["nA_max"] = nA.max()
                        scalars["nI_max"] = nI.max()

                scalar_groups = {}
                if self.cfg.laser.expose_results:
                    scalar_groups["P_lasers"] = {
                        f"L{i}": np.sum(laser.get_power(self.grid.X, self.grid.Y, t))
                        for i, laser in enumerate(self.lasers)
                    }

                self.visualizer.update(
                    t=t,
                    fields=fields,
                    scalars=scalars,
                    scalar_groups=scalar_groups,
                )

                self.next_viz_time += self.cfg.result.real_time_refresh_interval

            t += self.cfg.solver.dt
            self.state.t = t
