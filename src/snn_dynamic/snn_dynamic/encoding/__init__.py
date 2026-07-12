"""MNIST loading and power encoding."""

from snn_dynamic.encoding.downsample import load_and_downsample
from snn_dynamic.encoding.power_mapping import encode_powers

__all__ = ["encode_powers", "load_and_downsample"]
