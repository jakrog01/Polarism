from abc import ABC, abstractmethod

class AbstractLaser (ABC):
    @abstractmethod
    def P_space(self, X, Y): 
        raise NotImplementedError
    
    @abstractmethod
    def P_time(self, t):
        raise NotImplementedError
    
    def P(self, X, Y, dt):
        return self.P_space(X, Y) * self.P_time(dt)
