from polarism.reservoir.single_reservoir import SingleReservoir
from polarism.reservoir.double_reservoir import DoubleReservoir

def create_reservoir(reservoir_config, physics, grid):
    if reservoir_config.reservoir_type == "single":
        return SingleReservoir(reservoir_config, physics, grid)
    elif reservoir_config.reservoir_type == "double":
        return DoubleReservoir(reservoir_config, physics, grid)
    else:
        raise ValueError(f"Unknown reservoir type: {reservoir_config.reservoir_type}")
