from __future__ import annotations

available_boundary_conditions: dict[str, type] = {}


def register_absorption(name: str):
    def decorator(cls: type) -> type:
        available_boundary_conditions[name] = cls
        return cls

    return decorator
