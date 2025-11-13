import tyro
import polarism as ps

def main():
    cfg = tyro.cli(ps.Config)
    grid = ps.SimulationGrid2D(cfg.grid)
    boundry_condition_strategy = ps.BoundaryCondition(grid, cfg.boundry_condition, cfg.physics)
    potential = ps.create_potential(cfg.potential, grid)
    lasers = [ps.LaserFactory.create_laser(cfg.laser) for _ in range(cfg.laser.laser_count)]
    reservoir = ps.create_reservoir(cfg.reservoir, grid)

if __name__ == "__main__":
    main()
