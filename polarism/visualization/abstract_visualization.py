from abc import ABC, abstractmethod

class AbstractVisualization(ABC):
    @abstractmethod
    def plot(self, P, psi, nR, grid):
        pass