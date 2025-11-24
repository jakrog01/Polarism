from polarism.laser.laser_registy import available_lasers

class LaserFactory:
    @staticmethod
    def create_laser(laser_config):
        if laser_config.mode == "multiple":
            if laser_config.config_file is None:
                raise ValueError("config_file must be provided for multiple laser mode.")
            
            return LaserFactory._create_multiple_lasers(laser_config)
        
        return LaserFactory._create_single_laser(laser_config)


    @staticmethod
    def _create_single_laser(laser_config):
        cfg = normalize_config(laser_config)
        laser_type = cfg["laser_type"]
        laser_cls = available_lasers.get(laser_type)
        if laser_cls is None:
            raise ValueError(f"Unknown laser type: {laser_type}")
        return laser_cls(cfg)

    @staticmethod
    def _create_multiple_lasers(laser_config):
        cfg = normalize_config(laser_config)

        import yaml
        with open(cfg["config_file"], "r") as f:
            data = yaml.safe_load(f)

        lasers = []
        for item in data.get("lasers", []):
            cfg = normalize_config(item)
            laser_type = cfg["laser_type"]
            laser_cls = available_lasers[laser_type]
            lasers.append(laser_cls(cfg))
        return lasers
    
def normalize_config(cfg):
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return vars(cfg)
    raise TypeError("Unsupported laser config format")