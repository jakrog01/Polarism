import tyro
import polarism as ps

def main():
    cfg = tyro.cli(ps.config.Config)
    grid = ps.SimulationGrid2D(cfg.grid)

    boundry_condition_strategy = ps.BoundaryConditionFactory.create_boundary_condition_strategy(
        grid, cfg.boundry_condition,cfg.physics)
        
    potential = ps.PotentialDispatcher(cfg.potential)

if __name__ == "__main__":
    main()
