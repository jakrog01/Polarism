import tyro
from config.simulation_parameters import Config
from simulation.simulation_grid_2D import SimulationGrid2D as Grid2D
from simulation.boundry_conditions_strategies.boundry_condition_factory import BoundryConditionFactory
def main():
    cfg = tyro.cli(Config)
    grid = Grid2D(cfg.grid)
  
    boundry_condition_strategy = BoundryConditionFactory.create_boundry_condition_strategy(
        grid, cfg.boundry_condition,cfg.physics)

if __name__ == "__main__":
    main()
