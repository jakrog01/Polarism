import numpy as np
import matplotlib.pyplot as plt
from polarism.visualization.abstract_visualization import AbstractVisualization

class RealTimeVisualization(AbstractVisualization):
    def plot(self, P, psi, nR, grid):
        if getattr(self, '_interactive_fig', None) is None:
            self._init_interactive_plot(P, grid, nR, psi)

        try:
            self._im0.set_data(P)
            self._im1.set_data(nR)
            self._im2.set_data(np.abs(psi)**2)

            try:
                self._im0.set_clim(np.nanmin(P), np.nanmax(P))
            except Exception:
                pass
            try:
                self._im1.set_clim(np.nanmin(nR), np.nanmax(nR))
            except Exception:
                pass
            try:
                im2data = np.abs(psi)**2
                self._im2.set_clim(np.nanmin(im2data), np.nanmax(im2data))
            except Exception:
                pass
            if self._interactive_fig is not None:
                self._interactive_fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(P, extent=[grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()], origin='lower', cmap='inferno')
            axes[0].set_title("Laser pump $P(x,y)$")
            axes[1].imshow(nR, extent=[grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()], origin='lower', cmap='viridis')
            axes[1].set_title("nR")
            axes[2].imshow(np.abs(psi)**2, extent=[grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()], origin='lower', cmap='magma')
            axes[2].set_title("Condensate density $|\\psi|^2$")
            plt.tight_layout()
            plt.show()

    def _init_interactive_plot(self, P, grid, nR, psi):
        self._interactive_fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        extent = [grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()]

        self._im0 = axes[0].imshow(P, extent=extent, origin='lower', cmap='inferno')
        axes[0].set_title("Laser pump $P(x,y)$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        self._cb0 = self._interactive_fig.colorbar(self._im0, ax=axes[0], shrink=0.8)

        self._im1 = axes[1].imshow(nR, extent=extent, origin='lower', cmap='viridis')
        axes[1].set_title("nR")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        self._cb1 = self._interactive_fig.colorbar(self._im1, ax=axes[1], shrink=0.8)

        self._im2 = axes[2].imshow(np.abs(psi)**2, extent=extent, origin='lower', cmap='magma')
        axes[2].set_title("Condensate density $|\\psi|^2$")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        self._cb2 = self._interactive_fig.colorbar(self._im2, ax=axes[2], shrink=0.8)

        plt.tight_layout()
        plt.show(block=False)