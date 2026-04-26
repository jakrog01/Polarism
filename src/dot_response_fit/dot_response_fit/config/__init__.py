"""Config loading, validation, and building for the dot-response-fit pipeline."""

__all__ = [
    "load_config",
    "extract_mnist_cfg",
    "extract_encoding_cfg",
    "extract_reference_cfg",
    "extract_fit_cfg",
    "build_scenario_config",
    "build_scenario_lasers",
    "build_mnist_lasers",
]

_LOADER_EXPORTS = {
    "load_config",
    "extract_mnist_cfg",
    "extract_encoding_cfg",
    "extract_reference_cfg",
    "extract_fit_cfg",
}
_BUILDER_EXPORTS = {
    "build_scenario_config",
    "build_scenario_lasers",
    "build_mnist_lasers",
}


def __getattr__(name: str):
    """Load config helpers lazily so validation does not import NumPy/CuPy."""
    if name in _LOADER_EXPORTS:
        from dot_response_fit.config import loader

        return getattr(loader, name)
    if name in _BUILDER_EXPORTS:
        from dot_response_fit.config import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
