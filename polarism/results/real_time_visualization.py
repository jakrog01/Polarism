"""Live result plotting."""
from __future__ import annotations

import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from polarism.results.result_groups import Results2D, ResultScalar, ResultScalarGroup


class RealTimeVisualization:
    """Plot fields and scalars during a run."""
    _initialized: bool
    _closed: bool
    _t: list[float]
    _scalar_data: dict[str, list[float]]
    _group_data: dict[str, dict[str, list[float]]]
    _scalar_axes: dict[str, Axes]
    _group_axes: dict[str, Axes]
    _im: dict[str, AxesImage]
    _clim_fixed: dict[str, bool]
    _lines: dict[str, Any]
    _group_lines: dict[str, dict[str, Any]]

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
        """Set up live plots for the requested results."""
        self.fields_2d = fields_2d
        self.scalars = scalars
        self.scalar_groups = scalar_groups
        self.tmax = tmax
        self.extent = grid_extent
        self.pause_s = pause_s
        self._initialized = False
        self._closed = False
        self._t = []
        self._scalar_data = {s.name: [] for s in scalars}
        self._group_data = {
            g.name: {label: [] for label in g.labels} for g in scalar_groups
        }
        self._scalar_axes = {}
        self._group_axes = {}

    def _init_figure(self) -> None:
        """Create the matplotlib figure and artists."""
        n_fields = len(self.fields_2d)
        n_scalars = len(self.scalars) + len(self.scalar_groups)
        ncols = max(1, max(n_fields, n_scalars))
        self.fig, self.axes = plt.subplots(
            2, ncols, figsize=(5 * ncols, 8), squeeze=False
        )
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self._im = {}
        self._clim_fixed = {}
        self._colorbars = {}
        for i, field in enumerate(self.fields_2d):
            ax = self.axes[0, i]
            im = ax.imshow(
                np.zeros((10, 10)), extent=self.extent, origin="lower", cmap=field.cmap
            )
            ax.set_title(field.name)
            cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
            self._colorbars[field.name] = cbar
            if field.clim is not None:
                im.set_clim(field.clim[0], field.clim[1])
                self._clim_fixed[field.name] = True
            else:
                self._clim_fixed[field.name] = False
            self._im[field.name] = im
        for i in range(n_fields, ncols):
            self.axes[0, i].axis("off")
        self._lines = {}
        self._group_lines = {}
        col = 0
        for s in self.scalars:
            ax = self.axes[1, col]
            (line,) = ax.plot([], [], color=s.color)
            ax.set_title(s.name)
            ax.grid(True)
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
            ax.grid(True)
            self._group_axes[g.name] = ax
            col += 1
        for i in range(col, ncols):
            self.axes[1, i].axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        t: float,
        fields: dict[str, np.ndarray] | None = None,
        scalars: dict[str, float] | None = None,
        scalar_groups: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Update the plots with new fields and scalars."""
        if self._closed:
            return
        if not self._initialized:
            self._init_figure()
            self._initialized = True
        if "ipykernel" not in sys.modules and not plt.fignum_exists(self.fig.number):
            self._closed = True
            return
        self._t.append(t)
        if fields:
            for name, data in fields.items():
                if name in self._im:
                    self._im[name].set_data(data)
                    if not self._clim_fixed[name]:
                        vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
                        if np.isfinite(vmin) and np.isfinite(vmax):
                            if vmax > vmin:
                                self._im[name].set_clim(vmin, vmax)
                            else:
                                self._im[name].set_clim(
                                    vmin - abs(vmin) * 0.1 - 1e-30,
                                    vmax + abs(vmax) * 0.1 + 1e-30,
                                )
                            self._colorbars[name].update_normal(self._im[name])
        if scalars:
            for name, value in scalars.items():
                if name not in self._lines:
                    continue
                self._scalar_data[name].append(value)
                self._lines[name].set_data(self._t, self._scalar_data[name])
                ax = self._scalar_axes[name]
                ax.set_xlim(0, max(1e-5, t))
                ax.relim()
                ax.autoscale_view(scaley=True, scalex=False)
        if scalar_groups:
            for gname, values in scalar_groups.items():
                if gname not in self._group_lines:
                    continue
                for label, value in values.items():
                    if label in self._group_lines[gname]:
                        self._group_data[gname][label].append(value)
                        self._group_lines[gname][label].set_data(
                            self._t, self._group_data[gname][label]
                        )
                ax = self._group_axes[gname]
                ax.set_xlim(0, max(1e-5, t))
                ax.relim()
                ax.autoscale_view(scaley=True, scalex=False)
        try:
            if "ipykernel" in sys.modules:
                from IPython.display import clear_output, display

                self.fig.canvas.draw()
                clear_output(wait=True)
                display(self.fig)
            else:
                plt.draw()
                plt.pause(self.pause_s)
        except Exception:
            self._closed = True

    def _on_close(self, event) -> None:
        """Mark the window as closed."""
        self._closed = True
