from abc import ABC, abstractmethod


class AbstractReservoir(ABC):
    @abstractmethod
    def __init__(self, reservoir_config, physcis):
        pass

    @abstractmethod
    def step(self, dt, psi, Pxy):
        pass
