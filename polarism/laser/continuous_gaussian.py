from polarism.laser.abstract_laser import AbstractLaser
import numpy as np

class ContinuousGaussian(AbstractLaser):
    def __init__(self, P0, x0, y0, sigma):
        self.P0 = P0
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma

    def P_space(self, X, Y):
        r2 = (X-self.x0)**2 + (Y-self.y0)**2
        return self.P0*np.exp(-0.5*r2/self.sigma**2)
    
    def P_time(self, t): 
        return 1.0
