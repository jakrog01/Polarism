import sys
from typing import Union, Any

import numpy as np
import cupy as cp

from polarism.config.simulation_parameters import ComputeEngineParameters


class ComputeEngine:
    xp: Any
    use_gpu: bool
    config: ComputeEngineParameters | None

    def __init__(self):
        self.xp = np
        self.use_gpu = False
        self.config = None

    def configure(self, config: ComputeEngineParameters):
        self.config = config
        if self.config.use_gpu:
            try:
                import cupy as cp

                self.xp = cp
                self.use_gpu = True

            except ImportError:
                sys.stderr.write(
                    "CuPy is not installed. Falling back to CPU computation.\n"
                )
                self.xp = np
                self.use_gpu = False

    def to_gpu(self, array: Union[np.ndarray, Any]) -> Union[np.ndarray, Any]:
        return self.xp.asarray(array)

    def to_cpu(self, value: Union[np.ndarray, cp.ndarray]) -> np.ndarray:
        if self.use_gpu and isinstance(value, cp.ndarray):
            if hasattr(value, 'get'):
                return value.get()
            return np.array(value)
        
        if isinstance(value, (int, float, complex)):
            return np.array(value)

        return np.asarray(value)

compute_engine = ComputeEngine()
