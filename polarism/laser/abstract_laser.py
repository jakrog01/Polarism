from abc import ABC, abstractmethod

class AbstractLaser (ABC):
    def __init__(self, laser_config):
        self.P0 = laser_config["P0"]
        self.x0 = laser_config["x0"]
        self.y0 = laser_config["y0"]
        self.P = 0.0

    @abstractmethod
    def P_space(self, X, Y): 
        raise NotImplementedError
    
    @abstractmethod
    def P_time(self, t):
        raise NotImplementedError
    
    def get_pump_power(self, X, Y, t):
        self.P = self.P0 * self.P_space(X, Y) * self.P_time(t)
        return self.P
