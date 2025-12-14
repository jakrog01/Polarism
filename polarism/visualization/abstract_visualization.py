from abc import ABC, abstractmethod

class AbstractVisualization(ABC):
    @abstractmethod
    def plot(self, t, P, psi, nR, grid):
        pass