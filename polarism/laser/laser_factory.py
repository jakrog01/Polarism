from polarism.laser.laser_register import available_lasers

class LaserFactory:
    @staticmethod
    def create_laser(laser_config):
        laser_type = laser_config.type
        laser_cls = available_lasers.get(laser_type)
        if laser_cls is None:
            raise ValueError(f"Unknown laser type: {laser_type}")
        return laser_cls(laser_config)