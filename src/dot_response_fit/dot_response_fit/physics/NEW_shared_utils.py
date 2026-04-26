import numpy as np
import json
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

INITIAL_STATE = [1e-6, 0, 0]
POINTS_PER_PULSE = 15
SOLVER_PARAMS = {"rtol": 1e-6, "atol": 1e-7}


class AmplitudeCoding:
    def __init__(self, max_amp, pulse_width, separation, min_amp=0):
        self.max_amp = max_amp
        self.min_amp = min_amp
        self.pulse_width = pulse_width
        self.separation = separation

    def encode(self, image_flat):
        amps = self.min_amp + image_flat * (self.max_amp - self.min_amp)
        centers = np.arange(len(image_flat)) * self.separation
        return amps, centers


class SeparationCoding:
    def __init__(self, max_amp, pulse_width, min_separation, max_separation):
        self.max_amp = max_amp
        self.pulse_width = pulse_width
        self.min_separation = min_separation
        self.max_separation = max_separation

    def encode(self, image_flat):
        amps = np.ones_like(image_flat) * self.max_amp
        separations = self.min_separation + image_flat * (self.max_separation - self.min_separation)
        centers = np.zeros(len(image_flat))
        if len(image_flat) > 1:
            centers[1:] = np.cumsum(separations[:-1])
        return amps, centers


def build_coding(cfg):
    if cfg["name"] == "amplitude":
        return AmplitudeCoding(**cfg["params"])
    elif cfg["name"] == "separation":
        return SeparationCoding(**cfg["params"])
    raise ValueError(f"Unknown coding scheme: {cfg['name']}")


def rhs_factory(P, params):
    R, kappa = params['R'], params['kappa']
    gC, gR, gI = params['gammaC'], params['gammaR'], params['gammaI']

    def f(t, y):
        nc, nR, nI = y
        Pt = P(t)
        return [
            R * nR * nc - gC * nc + 1e-6,
            kappa * nI * nI - gR * nR - R * nR * nc,
            Pt - kappa * nI * nI - gI * nI
        ]

    return f


def generate_pump(amps, centers, width, t_eval):
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    inv = 1.0 / (2 * sigma ** 2)
    pump = np.zeros_like(t_eval)
    for a, c in zip(amps, centers):
        if a > 0:
            pump += a * np.exp(-(t_eval - c) ** 2 * inv)
    return interp1d(t_eval, pump, bounds_error=False, fill_value=0.0)


def simulate_image(image_flat, coding_cfg, ode_params):
    coding = build_coding(coding_cfg)
    amps, centers = coding.encode(image_flat)
    pulse_width = coding_cfg["params"]["pulse_width"]
    t_end = centers[-1] + (5 * pulse_width)
    dt = pulse_width / POINTS_PER_PULSE
    t_eval = np.arange(0, t_end, dt)

    P_interp = generate_pump(amps, centers, pulse_width, t_eval)
    rhs = rhs_factory(P_interp, ode_params)

    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), INITIAL_STATE,
        t_eval=t_eval, rtol=SOLVER_PARAMS['rtol'], atol=SOLVER_PARAMS['atol']
    )
    return t_eval, sol.y[0]


def save_config(filepath, config_dict):
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
