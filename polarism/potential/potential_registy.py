available_potentials = {}

def register_potential(name):
    def decorator(cls):
        available_potentials[name] = cls()
        return cls
    return decorator
