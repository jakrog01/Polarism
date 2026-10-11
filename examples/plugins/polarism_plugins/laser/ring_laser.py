from polarism.laser import register_laser
from polarism.laser.abstract_laser import AbstractLaser


@register_laser("example-ring")
class RingLaser(AbstractLaser):
    def __init__(self, laser_config, X, Y, precision="double"):
        super().__init__(laser_config, X, Y, precision)
        self.radius = laser_config.extra.get("ring_radius", 8.0)
        self.width = laser_config.sigma_space
        self._finalize_spatial_envelope(X, Y)

    def _amplitude(self, t):
        return self.P0

    def _P_space(self, X, Y):
        r = self.xp.sqrt((X - self.x0) ** 2 + (Y - self.y0) ** 2)
        return self.xp.exp(-0.5 * ((r - self.radius) / self.width) ** 2)

    def _P_time(self, t):
        return 1.0
