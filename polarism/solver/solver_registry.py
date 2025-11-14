available_solvers = {}

def register_solver(name):
    def decorator(cls):
        available_solvers[name] = cls
        return cls
    return decorator
