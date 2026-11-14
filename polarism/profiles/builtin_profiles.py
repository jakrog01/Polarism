"""Built-in analytic, file-backed and lattice spatial profiles."""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from polarism.profiles.profile_registry import register_profile


def _positive(spec, key, required=True):
    value = spec.get(key)
    if value is None and not required:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"profile.{key} must be a positive finite number")


def _radius_squared(grid, params):
    result = (grid.X - params["x0"]) ** 2
    if grid.ndim == 2:
        result = result + (grid.Y - params["y0"]) ** 2
    return result


@register_profile("uniform")
def uniform(grid, params, xp):
    """Return a constant field on 1D or 2D grids."""
    return xp.ones(grid.shape)


def _validate_gaussian(spec, ndim):
    _positive(spec, "sigma")


@register_profile("gaussian", params=("sigma",))
def gaussian(grid, params, xp):
    """Return exp(-r²/(2 sigma²))."""
    return xp.exp(-0.5 * _radius_squared(grid, params) / params["sigma"] ** 2)


gaussian.validate = _validate_gaussian


def _validate_supergaussian(spec, ndim):
    _positive(spec, "sigma")
    _positive(spec, "order")


@register_profile("supergaussian", params=("sigma", "order"))
def supergaussian(grid, params, xp):
    """Return exp(-0.5 (r²/sigma²)^order)."""
    return xp.exp(-0.5 * (_radius_squared(grid, params) / params["sigma"] ** 2) ** params["order"])


supergaussian.validate = _validate_supergaussian


def _validate_ring(spec, ndim):
    _positive(spec, "radius")
    _positive(spec, "width")


@register_profile("ring", params=("radius", "width"))
def ring(grid, params, xp):
    """Return a Gaussian radial cross-section centred at radius."""
    radius = xp.sqrt(_radius_squared(grid, params))
    return xp.exp(-0.5 * ((radius - params["radius"]) / params["width"]) ** 2)


ring.validate = _validate_ring


def _validate_file(spec, ndim):
    path = spec.get("path")
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("profile.path is required")
    suffix = Path(path).suffix.lower()
    if suffix not in {".npy", ".h5", ".hdf5", ".png"}:
        raise ValueError("profile.path must end in .npy, .h5, .hdf5 or .png")
    if suffix in {".h5", ".hdf5"} and not isinstance(spec.get("dataset"), str):
        raise ValueError("profile.dataset is required for HDF5 files")
    extent = spec.get("extent")
    if extent is not None:
        if not isinstance(extent, (list, tuple)) or len(extent) != 2 * ndim:
            raise ValueError(f"profile.extent must contain {2 * ndim} bounds")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in extent):
            raise ValueError("profile.extent must contain finite numbers")
        if extent[0] >= extent[1] or (ndim == 2 and extent[2] >= extent[3]):
            raise ValueError("profile.extent bounds must be increasing")


def _load_file(params):
    path = Path(params["path"])
    if not path.is_file():
        raise FileNotFoundError(f"profile.path {str(path)!r} does not exist")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            array = handle[params["dataset"]][...]
    else:
        from matplotlib.image import imread
        array = imread(path)
        if array.ndim == 3:
            array = array[..., :3].mean(axis=-1)
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float64) / np.iinfo(array.dtype).max
        array = np.clip(array, 0.0, 1.0)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("profile file must contain finite real values")
    return array


def _coordinate_indices(coordinate, lower, upper, length, xp):
    position = (coordinate - lower) * (length - 1) / (upper - lower)
    inside = (position >= 0) & (position <= length - 1)
    clipped = xp.clip(position, 0, length - 1)
    low = xp.floor(clipped).astype(xp.int64)
    high = xp.minimum(low + 1, length - 1)
    fraction = clipped - low
    return low, high, fraction, inside


@register_profile("file", params=("path", "dataset", "extent"))
def file_profile(grid, params, xp):
    """Interpolate a real file array at grid coordinates, with zero outside its extent."""
    source = _load_file(params)
    if source.ndim != grid.ndim or any(size < 2 for size in source.shape):
        raise ValueError("profile file dimensionality must match the grid and each axis needs two samples")
    data = xp.asarray(source, dtype=xp.float64)
    extent = params.get("extent")
    if extent is None:
        extent = [-grid.lx / 2, grid.lx / 2]
        if grid.ndim == 2:
            extent += [-grid.ly / 2, grid.ly / 2]
    ix0, ix1, fx, valid_x = _coordinate_indices(grid.X - params["x0"], extent[0], extent[1], source.shape[-1], xp)
    if grid.ndim == 1:
        return xp.where(valid_x, data[ix0] * (1 - fx) + data[ix1] * fx, 0.0)
    iy0, iy1, fy, valid_y = _coordinate_indices(grid.Y - params["y0"], extent[2], extent[3], source.shape[0], xp)
    lower = data[iy0, ix0] * (1 - fx) + data[iy0, ix1] * fx
    upper = data[iy1, ix0] * (1 - fx) + data[iy1, ix1] * fx
    return xp.where(valid_x & valid_y, lower * (1 - fy) + upper * fy, 0.0)


file_profile.validate = _validate_file


def _validate_lattice(spec, ndim):
    if spec.get("lattice") not in {"square", "triangular", "honeycomb", "kagome", "lieb"}:
        raise ValueError("profile.lattice must be square, triangular, honeycomb, kagome or lieb")
    _positive(spec, "constant")
    _positive(spec, "width")
    if spec.get("well", "gaussian") not in {"gaussian", "supergaussian"}:
        raise ValueError("profile.well must be gaussian or supergaussian")
    if spec.get("well", "gaussian") == "supergaussian":
        _positive(spec, "order")
    elif "order" in spec:
        raise ValueError("profile.order requires well: supergaussian")
    disorder = spec.get("disorder", 0.0)
    if isinstance(disorder, bool) or not isinstance(disorder, (int, float)) or not math.isfinite(disorder) or disorder < 0:
        raise ValueError("profile.disorder must be a nonnegative finite number")
    seed = spec.get("seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("profile.seed must be a nonnegative integer")
    rotation = spec.get("rotation", 0.0)
    if isinstance(rotation, bool) or not isinstance(rotation, (int, float)) or not math.isfinite(rotation):
        raise ValueError("profile.rotation must be a finite number")
    defects = spec.get("defects", [])
    if not isinstance(defects, list) or any(not isinstance(site, (list, tuple)) or len(site) != 3 or any(type(index) is not int for index in site) for site in defects):
        raise ValueError("profile.defects must be a list of [i, j, s] integer indices")
    counts = {"square": 1, "triangular": 1, "honeycomb": 2, "kagome": 3, "lieb": 3}
    if any(site[2] < 0 or site[2] >= counts[spec["lattice"]] for site in defects):
        raise ValueError("profile.defects contains an invalid sublattice index")


def _lattice_geometry(kind, a):
    """Return basis vectors and offsets; a is the nearest-neighbour length except for honeycomb.

    square: b1=(a,0), b2=(0,a); offsets (0,0); nearest a.
    triangular: b1=(a,0), b2=(a/2,sqrt(3)a/2); offsets (0,0); nearest a.
    honeycomb: b1=(a,0), b2=(a/2,sqrt(3)a/2); offsets (0,0), (a/2,a/(2sqrt(3))); nearest a/sqrt(3).
    kagome: b1=(2a,0), b2=(a,sqrt(3)a); offsets (0,0), (a,0), (a/2,sqrt(3)a/2); nearest a.
    lieb: b1=(2a,0), b2=(0,2a); offsets (0,0), (a,0), (0,a); nearest a.
    """
    h = math.sqrt(3)
    if kind == "square":
        return np.array([[a, 0], [0, a]]), np.array([[0, 0]])
    if kind == "triangular":
        return np.array([[a, 0], [a / 2, h * a / 2]]), np.array([[0, 0]])
    if kind == "honeycomb":
        return np.array([[a, 0], [a / 2, h * a / 2]]), np.array([[0, 0], [a / 2, a / (2 * h)]])
    if kind == "kagome":
        return np.array([[2 * a, 0], [a, h * a]]), np.array([[0, 0], [a, 0], [a / 2, h * a / 2]])
    return np.array([[2 * a, 0], [0, 2 * a]]), np.array([[0, 0], [a, 0], [0, a]])


def _lattice_sites(grid, params):
    """Yield ``(i, j, s, x, y)`` for sites covering the box plus one cell."""
    basis, offsets = _lattice_geometry(params["lattice"], params["constant"])
    theta = math.radians(params.get("rotation", 0.0))
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    basis = basis @ rotation.T
    offsets = offsets @ rotation.T
    origin = np.array([params["x0"], params["y0"]])
    corners = np.array([[x, y] for x in (-grid.lx / 2, grid.lx / 2) for y in (-grid.ly / 2, grid.ly / 2)])
    cell_coords = (corners - origin) @ np.linalg.inv(basis)
    lower = np.floor(cell_coords.min(axis=0)).astype(int) - 2
    upper = np.ceil(cell_coords.max(axis=0)).astype(int) + 2
    margin = max(np.linalg.norm(basis[0]), np.linalg.norm(basis[1]))
    defects = {tuple(site) for site in params.get("defects", [])}
    for i in range(lower[0], upper[0] + 1):
        for j in range(lower[1], upper[1] + 1):
            for s, offset in enumerate(offsets):
                if (i, j, s) in defects:
                    continue
                x, y = origin + i * basis[0] + j * basis[1] + offset
                if abs(x) <= grid.lx / 2 + margin and abs(y) <= grid.ly / 2 + margin:
                    yield i, j, s, float(x), float(y)


@register_profile("lattice", params=("lattice", "constant", "well", "width", "order", "disorder", "seed", "defects", "rotation"))
def lattice(grid, params, xp):
    """Sum unit-depth Gaussian or super-Gaussian wells on a 2D lattice."""
    result = xp.zeros(grid.shape, dtype=xp.float64)
    rng = np.random.default_rng(params.get("seed", 0))
    width = params["width"]
    for _, _, _, x, y in _lattice_sites(grid, params):
        depth = 1.0 + rng.normal(0.0, params.get("disorder", 0.0)) if params.get("disorder", 0.0) else 1.0
        scaled = ((grid.X - x) ** 2 + (grid.Y - y) ** 2) / width ** 2
        if params.get("well", "gaussian") == "supergaussian":
            scaled = scaled ** params["order"]
        result += depth * xp.exp(-0.5 * scaled)
    return result


lattice.validate = _validate_lattice
