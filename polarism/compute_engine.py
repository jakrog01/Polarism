"""Backend selection for NumPy and CuPy arrays."""
import sys
from typing import Any, Union

import numpy as np

from polarism.config.simulation_parameters import ComputeEngineParameters

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class ComputeEngine:
    """Track the active NumPy or CuPy backend."""
    xp: Any
    use_gpu: bool
    config: ComputeEngineParameters | None

    def __init__(self):
        """Set the default backend to NumPy."""
        self.xp = np
        self.use_gpu = False
        self.config = None

    def configure(self, config: ComputeEngineParameters):
        """Configure the compute backend."""
        self.config = config
        if self.config.use_gpu:
            if CUPY_AVAILABLE:
                self.xp = cp
                self.use_gpu = True
            else:
                sys.stderr.write(
                    "CuPy is not installed. Falling back to CPU computation.\n"
                )
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
            self.use_gpu = False

    def to_gpu(self, array: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        """Move an array to the active backend."""
        return self.xp.asarray(array)

    def to_cpu(self, value: Any) -> np.ndarray:

        """Move a value to NumPy memory."""
        if self.use_gpu and CUPY_AVAILABLE:
            if isinstance(value, cp.ndarray):
                return value.get()
            if hasattr(value, "get"):
                return np.asarray(value.get())

        if isinstance(value, (int, float, complex)):
            return np.array(value)

        return np.asarray(value)


compute_engine = ComputeEngine()
