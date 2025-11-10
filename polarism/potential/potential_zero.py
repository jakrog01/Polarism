import numpy as np
from .potential_registy import register_potential

@register_potential("zero")
class PotentialZero:
    def __call__(self, X):
        return np.zeros_like(X)
