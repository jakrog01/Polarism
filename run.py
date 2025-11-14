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
    solver = ps.create_solver(state, cfg, grid, potential, lasers, reservoir, boundry_condition_strategy)


if __name__ == "__main__":
    main()
