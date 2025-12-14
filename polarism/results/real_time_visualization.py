import matplotlib.pyplot as plt
import numpy as np

from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup


class RealTimeVisualization:
    def __init__(
        self,
        fields_2d: list[Results2D],
        scalars: list[ResultScalar],
        scalar_groups: list[ResultScalarGroup],
        tmax: float,
        grid_extent: list[float],
        pause_s: float = 0.001,
    ):
        self.fields_2d = fields_2d
        self.scalars = scalars
        self.scalar_groups = scalar_groups
        self.tmax = tmax
        self.extent = grid_extent
        self.pause_s = pause_s

        self._initialized = False
        self._t = []

        self._scalar_data = {s.name: [] for s in scalars}
        self._group_data = {
            g.name: {label: [] for label in g.labels} for g in scalar_groups
        }

        self._scalar_axes = {}
        self._group_axes = {}

    def update(self, t, fields=None, scalars=None, scalar_groups=None):
        if not self._initialized:
            self._init_figure()
            self._initialized = True

        self._t.append(t)
        frame_idx = len(self._t)

        if fields:
            for name, data in fields.items():
                if name in self._im:
                    self._im[name].set_data(data)
                    if self._clim_fixed[name] is False:
                        vmin = float(np.nanmin(data))
                        vmax = float(np.nanmax(data))
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                            self._im[name].set_clim(vmin, vmax)

        if scalars:
            for name, value in scalars.items():
                if name not in self._lines:
                    continue
                self._scalar_data[name].append(value)
                line = self._lines[name]
                line.set_data(self._t, self._scalar_data[name])
                ax = self._scalar_axes[name]
                ax.relim()
                ax.autoscale_view(scaley=True)

        if scalar_groups:
            for gname, values in scalar_groups.items():
                if gname not in self._group_lines:
                    continue
                for label, value in values.items():
                    if label not in self._group_lines[gname]:
                        continue
                    self._group_data[gname][label].append(value)
                    line = self._group_lines[gname][label]
                    line.set_data(self._t, self._group_data[gname][label])
                ax = self._group_axes[gname]
                ax.relim()
                ax.autoscale_view(scaley=True)

        plt.draw()
        plt.pause(self.pause_s)

    def _init_figure(self):
        ncols = len(self.fields_2d)
        self.fig, self.axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

        self._im = {}
        self._clim_fixed = {}

        for i, field in enumerate(self.fields_2d):
            ax = self.axes[0, i]
            im = ax.imshow(
                np.zeros((10, 10)),
                extent=self.extent,
                origin="lower",
                cmap=field.cmap,
            )
            ax.set_title(field.name)
            self.fig.colorbar(im, ax=ax, shrink=0.8)

            if field.clim is not None:
                im.set_clim(*field.clim)
                self._clim_fixed[field.name] = True
            else:
                self._clim_fixed[field.name] = False

            self._im[field.name] = im

        self._lines = {}
        self._group_lines = {}

        for ax in self.axes[1, :]:
            ax.set_xlim(0, self.tmax)
            ax.grid(True)

        col = 0
        for s in self.scalars:
            ax = self.axes[1, col]
            (line,) = ax.plot([], [], color=s.color)
            ax.set_title(s.name)
            self._lines[s.name] = line
            self._scalar_axes[s.name] = ax
            col += 1

        for g in self.scalar_groups:
            ax = self.axes[1, col]
            self._group_lines[g.name] = {}
            for label in g.labels:
                (line,) = ax.plot([], [], label=label)
                self._group_lines[g.name][label] = line
            ax.legend()
            ax.set_title(g.name)
            self._group_axes[g.name] = ax
            col += 1

        plt.tight_layout()
        plt.show(block=False)
