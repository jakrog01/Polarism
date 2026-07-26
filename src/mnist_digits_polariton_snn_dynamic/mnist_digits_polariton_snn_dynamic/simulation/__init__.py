"""Simulation helpers for the dynamic SNN pipeline."""

from typing import Any

__all__ = [
    "GridLatticeGeometry",
    "GeometryProvider",
    "SharedSimResources",
    "build_lasers",
    "simulate_one_image",
]


def __getattr__(name: str) -> Any:
    if name in {"GeometryProvider", "GridLatticeGeometry"}:
        from mnist_digits_polariton_snn_dynamic.simulation.geometry import GeometryProvider, GridLatticeGeometry

        return {
            "GeometryProvider": GeometryProvider,
            "GridLatticeGeometry": GridLatticeGeometry,
        }[name]
    if name == "build_lasers":
        from mnist_digits_polariton_snn_dynamic.simulation.lasers import build_lasers

        return build_lasers
    if name == "SharedSimResources":
        from mnist_digits_polariton_snn_dynamic.simulation.resources import SharedSimResources

        return SharedSimResources
    if name == "simulate_one_image":
        from mnist_digits_polariton_snn_dynamic.simulation.runner import simulate_one_image

        return simulate_one_image
    raise AttributeError(name)
