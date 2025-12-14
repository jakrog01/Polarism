import numpy as np

from polarism.laser.abstract_laser import AbstractLaser
from polarism.laser.laser_registy import register_laser


@register_laser("continuous-gaussian")
class ContinuousGaussian(AbstractLaser):
    def __init__(self, laser_config):
        super().__init__(laser_config)
        self.sigma_space = laser_config["sigma_space"]

    def P_space(self, X, Y):
        r2 = (X - self.x0) ** 2 + (Y - self.y0) ** 2
        return np.exp(-0.5 * r2 / self.sigma_space**2)

    def P_time(self, t):
        return 1.0
