import tyro
import polarism as ps

def main():
    cfg = tyro.cli(ps.Config)
    grid = ps.SimulationGrid2D(cfg.grid)
    boundry_condition_strategy = ps.BoundaryCondition(grid, cfg.boundry_condition, cfg.physics)
    potential = ps.create_potential(cfg.potential, grid)
    laser = ps.LaserFactory.create_laser(cfg.laser)
    pass

if __name__ == "__main__":
    main()
