from polarism.solver.solver_registry import available_solvers

def create_solver(state, config, grid, potential, laser, reservoir, boundary_condition, visualizer):
    if config.solver.method not in available_solvers:
        raise ValueError(
            f"Unknown boundry conditions: '{config.solver.method}'. "
            f"Available: {list(available_solvers.keys())}"
        )
    
    return available_solvers[config.solver.method](state, config, grid, potential, laser, reservoir, boundary_condition, visualizer)