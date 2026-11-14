"""Built-in scalar temporal envelopes."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from polarism.envelopes.envelope_registry import register_envelope


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"envelope.{name} must be a {'positive ' if positive else ''}finite number")
    return float(value)


def _constant(t):
    return 1.0


_constant.time_independent = True


@register_envelope("constant")
def constant(spec):
    """Return one at all times."""
    return _constant


@register_envelope("step", params=("t0",))
def step(spec):
    """Return zero before t0 and one at and after t0."""
    t0 = _number(spec.get("t0"), "t0")
    return lambda t: 0.0 if t < t0 else 1.0


@register_envelope("gaussian-pulses", params=("center", "sigma", "count", "separation", "cutoff_sigma"))
def gaussian_pulses(spec):
    """Return a sum of truncated unit-height Gaussian pulses."""
    center = _number(spec.get("center"), "center")
    sigma = _number(spec.get("sigma"), "sigma", positive=True)
    count = spec.get("count", 1)
    if type(count) is not int or count < 1:
        raise ValueError("envelope.count must be a positive integer")
    separation = _number(spec.get("separation", 0.0), "separation")
    if count > 1 and separation <= 0:
        raise ValueError("envelope.separation must be positive when count > 1")
    cutoff = _number(spec.get("cutoff_sigma", 3.0), "cutoff_sigma", positive=True)
    centers = tuple(center + i * separation for i in range(count))

    def evaluate(t):
        total = 0.0
        for pulse_center in centers:
            distance = (t - pulse_center) / sigma
            if abs(distance) <= cutoff:
                total += math.exp(-0.5 * distance ** 2)
        return total

    return evaluate


def _data_rows(spec):
    has_path = "path" in spec
    has_inline = "times" in spec or "values" in spec
    if has_path == has_inline:
        raise ValueError("envelope.data requires either path or times and values")
    if has_path:
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"envelope.path {str(path)!r} does not exist")
        if path.suffix.lower() == ".npy":
            data = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as handle:
                data = np.asarray(list(csv.reader(handle)), dtype=float)
        else:
            raise ValueError("envelope.path must end in .npy or .csv")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("envelope data file must have shape (N, 2)")
        return data[:, 0], data[:, 1]
    return spec.get("times"), spec.get("values")


@register_envelope("data", params=("times", "values", "path", "hold"))
def data(spec):
    """Interpolate tabulated times and values; return zero outside by default."""
    times, values = _data_rows(spec)
    if not isinstance(times, (list, tuple, np.ndarray)) or not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("envelope.times and envelope.values must be sequences")
    times = np.asarray(times)
    values = np.asarray(values)
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values) or len(times) < 2:
        raise ValueError("envelope times and values must have equal lengths of at least two")
    if times.dtype.kind not in "fiu" or values.dtype.kind not in "fiu" or not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("envelope times and values must be finite real numbers")
    if not np.all(np.diff(times) > 0):
        raise ValueError("envelope.times must be strictly increasing")
    hold = spec.get("hold", False)
    if type(hold) is not bool:
        raise ValueError("envelope.hold must be a boolean")
    times = times.astype(float)
    values = values.astype(float)
    left = values[0] if hold else 0.0
    right = values[-1] if hold else 0.0
    return lambda t: float(np.interp(t, times, values, left=left, right=right))
