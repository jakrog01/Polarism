available_lasers = {}


def register_laser(name: str):
    def decorator(cls):
        available_lasers[name] = cls
        return cls

    return decorator
