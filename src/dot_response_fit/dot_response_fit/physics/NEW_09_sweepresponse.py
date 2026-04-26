import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider

t_start, t_end = 0.0, 400.0
t_eval = np.linspace(t_start, t_end, 1000)
amp_start, amp_end, num_amps = 0.0, 160.0, 60
mu, width = 20.0, 5.0
sigma = width / (2 * np.sqrt(2 * np.log(2)))
p = {"kappa": 0.05, "gammaR": 0.005, "gammaI": 0.001, "gammaC": 0.0833, "R": 0.0006}


def gaussian_pulse(t, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def ode_system(t, y, amp, mu, sigma, p):
    nr, ni, nc = y
    pulse = gaussian_pulse(t, amp, mu, sigma)
    dnr_dt = p["kappa"] * ni ** 2 - p['gammaR'] * nr - p["R"] * nr * nc
    dni_dt = pulse - p["kappa"] * ni ** 2 - p["gammaI"] * ni
    dnc_dt = p["R"] * nr * nc - p["gammaC"] * nc + 1e-6
    return [dnr_dt, dni_dt, dnc_dt]


amplitudes = np.linspace(amp_start, amp_end, num_amps)
nc_integrals, nc_trajectories, pulse_trajectories = [], [], []

for amp in amplitudes:
    sol = solve_ivp(ode_system, (t_start, t_end), [0.0, 0.0, 1e-6],
                    args=(amp, mu, sigma, p), t_eval=t_eval)
    nc_trajectories.append(sol.y[2])
    pulse_trajectories.append(gaussian_pulse(t_eval, amp, mu, sigma))
    nc_integrals.append(np.trapz(sol.y[2], t_eval))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
plt.subplots_adjust(bottom=0.2, hspace=0.4)

ax1.plot(amplitudes, nc_integrals, 'b-')
ax1.set_title("Integral of $n_c(t)$ vs Amplitude")
ax1.grid(True)

init_idx = num_amps // 2
l_pulse, = ax2.plot(t_eval, pulse_trajectories[init_idx], 'r--', alpha=0.5, label='Pulse')
l_nc, = ax2.plot(t_eval, nc_trajectories[init_idx], 'g-', lw=2, label='$n_c(t)$')

idx_max = np.argmax(nc_trajectories[init_idx])
v_line = ax2.axvline(t_eval[idx_max], color='gray', linestyle=':', alpha=0.7)
h_line = ax2.axhline(nc_trajectories[init_idx][idx_max], color='gray', linestyle=':', alpha=0.7)
txt = ax2.text(0.05, 1.12, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.5))

ax2.legend()
ax2.grid(True)

ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 'Amp', amp_start, amp_end, valinit=amplitudes[init_idx])


def update(val):
    idx = (np.abs(amplitudes - slider.val)).argmin()
    y_data = nc_trajectories[idx]
    imax = np.argmax(y_data)
    t_max, val_max = t_eval[imax], y_data[imax]

    l_pulse.set_ydata(pulse_trajectories[idx])
    l_nc.set_ydata(y_data)
    v_line.set_xdata([t_max, t_max])
    h_line.set_ydata([val_max, val_max])

    ax2.set_title(f"Dynamics (Amp: {amplitudes[idx]:.1f})")
    txt.set_text(f"Max $n_c$: {val_max:.4f}\nTime: {t_max:.2f}")

    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw_idle()


update(amplitudes[init_idx])
slider.on_changed(update)
plt.show()
