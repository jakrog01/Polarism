import tyro
from config.simulation_parameters import Config
from simulation.simulation_grid_2D import SimulationGrid2D as Grid2D
from simulation.boundry_conditions_strategies.absorption_mask_strategy import AbsorptionMaskStrategy
from simulation.boundry_conditions_strategies.absorption_perturbation_strategy import AbsorptionPerturbationStrategy
def main():
    cfg = tyro.cli(Config)
    grid = Grid2D(cfg.grid)
  
    if cfg.boundry_condition.absorption == "absorbtion-mask":
        absorption_strategy = AbsorptionMaskStrategy(grid, cfg.boundry_condition, cfg.physics)
    elif cfg.boundry_condition.absorption == "absorbtion-perturbation":
        absorption_strategy = AbsorptionPerturbationStrategy(grid, cfg.boundry_condition, cfg.physics)
    else:
        raise ValueError(f"Unknown boundary condition: {cfg.boundry_condition.boundary_condition}") 

if __name__ == "__main__":
    main()