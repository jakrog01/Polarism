from .potential_registy import available_potentials
from ..config.simulation_parameters import PotentialParameters

class PotentialDispatcher:
    def __init__(self, potential_config: PotentialParameters):
        if potential_config.potential_type not in available_potentials:
            raise ValueError(
                f"Nieznany potencjał: '{potential_config.potential_type}'. "
                f"Dostępne: {list(available_potentials.keys())}"
            )
        self.potential = available_potentials[potential_config.potential_type]

    def get_potential(self, X):
        return self.potential(X)
