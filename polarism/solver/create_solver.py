from polarism.solver.solver_registry import available_solvers

def create_solver(config, grid):
    if config.solver.method not in available_solvers:
        raise ValueError(
            f"Unknown boundry conditions: '{config.solver.method}'. "
            f"Available: {list(available_solvers.keys())}"
        )
    
    return available_solvers[config.solver.method](config, grid)