import numpy as np
import matplotlib.pyplot as plt

class RealTimeVisualization():
    def __init__(self):
        self._interactive_fig = None
        self._t_history = []
        self._P_history = []

    def plot(self, t, P, psi, nR, grid):
        if self._interactive_fig is None:
            self._init_interactive_plot(P, grid, nR, psi)

        self._t_history.append(t)
        self._P_history.append(np.nanmax(P))

        try:
            self._im0.set_data(P)
            self._im1.set_data(nR)
            self._im2.set_data(np.abs(psi)**2)

            self._im0.set_clim(np.nanmin(P), np.nanmax(P))
            self._im1.set_clim(np.nanmin(nR), np.nanmax(nR))

            im2data = np.abs(psi)**2
            self._im2.set_clim(np.nanmin(im2data), np.nanmax(im2data))

            self._lineP.set_data(self._t_history, self._P_history)
            self._axP.relim()
            self._axP.autoscale_view()

            self._interactive_fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            extent = [grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()]

            axes[0].imshow(P, extent=extent, origin="lower", cmap="inferno")
            axes[1].imshow(nR, extent=extent, origin="lower", cmap="viridis")
            axes[2].imshow(np.abs(psi)**2, extent=extent, origin="lower", cmap="magma")
            axes[3].plot(self._t_history, self._P_history, color="red")

            plt.tight_layout()
            plt.show()

    def _init_interactive_plot(self, P, grid, nR, psi):
        self._interactive_fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        extent = [grid.X.min(), grid.X.max(), grid.Y.min(), grid.Y.max()]

        self._im0 = axes[0].imshow(P, extent=extent, origin="lower", cmap="inferno")
        self._im1 = axes[1].imshow(nR, extent=extent, origin="lower", cmap="viridis")
        self._im2 = axes[2].imshow(np.abs(psi)**2, extent=extent, origin="lower", cmap="magma")

        self._axP = axes[3]
        self._lineP, = self._axP.plot([], [], color="red")
        self._axP.set_xlabel("t")
        self._axP.set_ylabel("max P")
        self._axP.set_title("P(t)")
        self._axP.grid(True)

        self._interactive_fig.colorbar(self._im0, ax=axes[0], shrink=0.8)
        self._interactive_fig.colorbar(self._im1, ax=axes[1], shrink=0.8)
        self._interactive_fig.colorbar(self._im2, ax=axes[2], shrink=0.8)

        plt.tight_layout()
        plt.show(block=False)
