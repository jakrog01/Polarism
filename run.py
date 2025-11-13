import tyro
import polarism as ps

def main():
    cfg = tyro.cli(ps.config.Config)
    grid = ps.SimulationGrid2D(cfg.grid)
    boundry_condition_strategy = ps.BoundaryCondition(grid, cfg.boundry_condition, cfg.physics)
    potential = ps.create_potential(cfg.potential, grid)

if __name__ == "__main__":
    main()
