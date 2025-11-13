from polarism.config.simulation_parameters import LaserParameters
from polarism.laser.laser_register import register_laser
from polarism.laser.abstract_laser import AbstractLaser
import numpy as np

@register_laser("continuous-gaussian")
class ContinuousGaussian(AbstractLaser):
    def __init__(self, laser_config: LaserParameters):
        self.P0 = laser_config.P0
        self.x0 = laser_config.x0
        self.y0 = laser_config.y0
        self.sigma = laser_config.sigma

    def P_space(self, X, Y):
        r2 = (X-self.x0)**2 + (Y-self.y0)**2
        return self.P0*np.exp(-0.5*r2/self.sigma**2)
    
    def P_time(self, t): 
        return 1.0
