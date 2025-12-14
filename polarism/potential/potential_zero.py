import numpy as np

from polarism.potential.potential_registy import register_potential


@register_potential("zero")
def potential_zero(X):
    return np.zeros_like(X)
