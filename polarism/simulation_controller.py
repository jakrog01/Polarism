import tyro
import numpy as np
import polarism as ps

class SimulationController:
    def __init__(self):
        self.cfg = tyro.cli(ps.Config)
        self.grid = ps.SimulationGrid2D(self.cfg.grid)
        self.boundry_condition = ps.BoundaryCondition(self.grid, self.cfg.boundry_condition, self.cfg.physics)
        self.potential = ps.create_potential(self.cfg.potential, self.grid)
        self.lasers = ps.LaserFactory.create_laser(self.cfg.laser)
        self.reservoir = ps.create_reservoir(self.cfg.reservoir, self.cfg.physics, self.grid)
        self.state = ps.SimulationState(self.grid)
        self.visualizer = ps.RealTimeVisualization()
        self.solver = ps.create_solver(self.cfg, self.grid)
        self.potential += self.boundry_condition.before_step_action()
        self.steps_count = 0

    def run(self):
        while self.state.t < self.cfg.solver.total_time:

            P_total = np.zeros_like(self.grid.X)
            for laser in self.lasers:
                P_total += laser.get_power(self.grid.X, self.grid.Y, self.state.t)
            
            self.solver.step(self.potential, P_total, self.reservoir, self.boundry_condition, self.state)

            if (self.steps_count % 100) == 0:
                self.visualizer.plot(self.state.t, P_total, self.state.psi, self.reservoir.get_reservoir_density(), self.grid)
                
            self.steps_count += 1
            self.state.t += self.cfg.solver.dt
            print(f"Simulation time: {self.state.t:.3f} / {self.cfg.solver.total_time:.3f}", end='\r')