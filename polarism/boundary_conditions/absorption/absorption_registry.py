available_boundry_conditions = {}

def register_absorption(name: str):
    def decorator(cls):
        available_boundry_conditions[name] = cls
        return cls
    return decorator
