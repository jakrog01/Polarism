import tyro
from config.simulation_parameters import Config
from simulation.simulation_grid_2D import SimulationGrid2D as Grid2D

def main():
    cfg = tyro.cli(Config)
    grid = Grid2D(cfg.grid)

if __name__ == "__main__":
    main()