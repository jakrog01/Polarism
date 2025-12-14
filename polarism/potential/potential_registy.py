available_potentials = {}


def register_potential(name):
    def decorator(func):
        available_potentials[name] = func
        return func

    return decorator
