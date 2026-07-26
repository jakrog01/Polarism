"""MNIST loading and power encoding."""

from typing import Any

__all__ = ["Encoder", "LinearPixelEncoder", "load_and_downsample"]


def __getattr__(name: str) -> Any:
    if name in {"Encoder", "LinearPixelEncoder"}:
        from mnist_digits_polariton_snn_dynamic.encoding.base import Encoder, LinearPixelEncoder

        return {"Encoder": Encoder, "LinearPixelEncoder": LinearPixelEncoder}[name]
    if name == "load_and_downsample":
        from mnist_digits_polariton_snn_dynamic.encoding.downsample import load_and_downsample

        return load_and_downsample
    raise AttributeError(name)
