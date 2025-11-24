import tyro
import polarism as ps

def main():
    cfg = tyro.cli(ps.Config)
    grid = ps.SimulationGrid2D(cfg.grid)
    boundry_condition_strategy = ps.BoundaryCondition(grid, cfg.boundry_condition, cfg.physics)
    potential = ps.create_potential(cfg.potential, grid)
    lasers = ps.LaserFactory.create_laser(cfg.laser)
    reservoir = ps.create_reservoir(cfg.reservoir, cfg.physics)
    state = ps.SimulationState(grid)
    visualizer = ps.RealTimeVisualization()
    solver = ps.create_solver(state, cfg, grid, potential, lasers, reservoir, boundry_condition_strategy, visualizer)
    
    while state.t < cfg.solver.total_time:
        solver.step()
        print(f"Simulation time: {state.t:.3f} / {cfg.solver.total_time:.3f}", end='\r')

if __name__ == "__main__":
    main()
