from polarism.grid import register_grid
from polarism.grid.periodic_simulation_grid_2d import PeriodicSimulationGrid2D


@register_grid("example-labelled-periodic")
class LabelledPeriodicGrid(PeriodicSimulationGrid2D):
    boundary = "periodic"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.label = cfg.extra.get("label", "example")
