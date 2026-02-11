from __future__ import annotations

import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup

MAX_PLOT_POINTS = 2000


class RealTimeVisualization:
    _initialized: bool
    _closed: bool
    _t: deque[float]
    _scalar_data: dict[str, deque[float]]
    _group_data: dict[str, dict[str, deque[float]]]
    _scalar_axes: dict[str, Axes]
    _group_axes: dict[str, Axes]
    _im: dict[str, AxesImage]
    _clim_fixed: dict[str, bool]

    fields_2d: list[Results2D]
    scalars: list[ResultScalar]
    scalar_groups: list[ResultScalarGroup]
    tmax: float
    extent: list[float]
    pause_s: float
    axes: np.ndarray
    fig: Figure

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
        self._closed = False
        self._t = deque(maxlen=MAX_PLOT_POINTS)

        self._scalar_data = {s.name: deque(maxlen=MAX_PLOT_POINTS) for s in scalars}
        self._group_data = {
            g.name: {label: deque(maxlen=MAX_PLOT_POINTS) for label in g.labels}
            for g in scalar_groups
        }

        self._scalar_axes = {}
        self._group_axes = {}

    def update(
        self,
        t: float,
        fields: dict[str, np.ndarray] | None = None,
        scalars: dict[str, float] | None = None,
        scalar_groups: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if self._closed:
            return

        if not self._initialized:
            self._init_figure()
            self._initialized = True

        if not plt.fignum_exists(self.fig.number):
            self._closed = True
            return

        self._t.append(t)
        t_list = list(self._t)

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
                line.set_data(t_list, list(self._scalar_data[name]))
                ax = self._scalar_axes[name]
                ax.relim()
                ax.autoscale_view(scaley=True)

                right = max(t_list) if t_list else 0.0
                if right <= 0.0:
                    ax.set_xlim(0, self.tmax)
                else:
                    ax.set_xlim(0, right)

                nticks = min(6, max(2, len(t_list)))
                xticks = np.linspace(0, ax.get_xlim()[1], nticks)
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{v:.2f}" for v in xticks])

        if scalar_groups:
            for gname, values in scalar_groups.items():
                if gname not in self._group_lines:
                    continue
                for label, value in values.items():
                    if label not in self._group_lines[gname]:
                        continue
                    self._group_data[gname][label].append(value)
                    line = self._group_lines[gname][label]
                    line.set_data(t_list, list(self._group_data[gname][label]))
                ax = self._group_axes[gname]
                ax.relim()
                ax.autoscale_view(scaley=True)

                right = max(t_list) if t_list else 0.0
                if right <= 0.0:
                    ax.set_xlim(0, self.tmax)
                else:
                    ax.set_xlim(0, right)

                nticks = min(6, max(2, len(t_list)))
                xticks = np.linspace(0, ax.get_xlim()[1], nticks)
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{v:.2f}" for v in xticks])

        try:
            plt.draw()
            plt.pause(self.pause_s)
        except Exception:
            self._closed = True

    def _init_figure(self) -> None:
        ncols = len(self.fields_2d)
        self.fig, self.axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

        self.fig.canvas.mpl_connect("close_event", self._on_close)

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

    def _on_close(self, event) -> None:
        self._closed = True
        sys.stderr.write("Visualization window closed.\n")
        sys.exit(2)
