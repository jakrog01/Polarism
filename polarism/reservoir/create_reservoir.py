from polarism.reservoir.single_reservoir import SingleReservoir

def create_reservoir(reservoir_config, physics):
    if reservoir_config.reservoir_type == "single":
        return SingleReservoir(reservoir_config, physics)
    else:
        raise ValueError(f"Unknown reservoir type: {reservoir_config.reservoir_type}")
