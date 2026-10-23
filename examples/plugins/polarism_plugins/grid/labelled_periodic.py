from polarism.grid import register_grid
from polarism.grid.periodic_grid import PeriodicGrid


@register_grid("example-labelled-periodic")
class LabelledPeriodicGrid(PeriodicGrid):
    boundary = "periodic"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.label = cfg.extra.get("label", "example")
