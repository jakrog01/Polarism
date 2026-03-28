import argparse
import json
import os
import shutil
from multiprocessing import Pool

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec


def _find_ffmpeg():
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
        return True
    except (ImportError, FileNotFoundError, subprocess.CalledProcessError):
        pass
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return False


HAS_FFMPEG = _find_ffmpeg()
GIF_MAX_FRAMES = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
ROUTINES = ["single", "double", "cross"]
FIELD_SPECS = {
    "psi_sq": {
        "source": "psi",
        "label": r"$|\psi|^2$",
        "cmap": "magma",
        "transform": "abs2",
    },
    "nI": {"source": "nI", "label": r"$n_I$", "cmap": "plasma", "transform": None},
    "nA": {"source": "nA", "label": r"$n_A$", "cmap": "viridis", "transform": None},
    "Pump": {
        "source": "Pump",
        "label": "Pump",
        "cmap": "inferno",
        "transform": None,
        "norm": "power",
    },
}
SCALAR_MAP = {"psi_sq": "psi_sq_max", "nI": "nI_max", "nA": "nA_max", "Pump": "P_max"}
SNAPSHOT_COUNT = 5
ANIM_FPS = 8
ANIM_GIF_FPS = 5
ANIM_TARGET_SECONDS = 60
ANIM_DPI = 150
PLOT_DPI = 200
PUMP_NORM_GAMMA = 0.3

ROUTINE_STYLES = {
    "single": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.5},
    "double": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.5},
    "cross": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.5},
}

DATA_DIR = SCRIPT_DIR


def routine_dir(routine):
    return os.path.join(RESULTS_DIR, routine)


def load_grid_extent():
    for d in (DATA_DIR, SCRIPT_DIR):
        cfg_path = os.path.join(d, "optimal_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            lx, ly = cfg["lx"], cfg["ly"]
            return [-lx / 2, lx / 2, -ly / 2, ly / 2]
    raise FileNotFoundError("optimal_config.json not found")


def open_h5(routine):
    return h5py.File(os.path.join(DATA_DIR, f"{routine}.h5"), "r")


def load_sorted_order(h5):
    time = h5["time"][:]
    return np.argsort(time, kind="stable")


def _read_field_frame(h5, spec, idx):
    raw = h5[f"fields/{spec['source']}"][idx]
    if spec["transform"] == "abs2":
        return np.abs(raw) ** 2
    return raw


def pick_indices(n, count=SNAPSHOT_COUNT):
    if n <= count:
        return list(range(n))
    return [int(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_norm(spec, vmin, vmax):
    if spec.get("norm") == "power":
        return PowerNorm(gamma=PUMP_NORM_GAMMA, vmin=max(vmin, 1e-12), vmax=vmax)
    return None


def generate_field_png(routine, field_key, extent):
    spec = FIELD_SPECS[field_key]

    with open_h5(routine) as h5:
        sort_order = load_sorted_order(h5)
        time_sorted = h5["time"][:][sort_order]
        n_frames = time_sorted.shape[0]

        snap_logical = pick_indices(n_frames)
        snap_physical = [int(sort_order[i]) for i in snap_logical]

        scalar_key = SCALAR_MAP.get(field_key)
        has_scalar = scalar_key is not None and f"scalars/{scalar_key}" in h5

        per_laser_keys = []
        if field_key == "Pump":
            i = 0
            while f"scalars/P_max_{i}" in h5:
                per_laser_keys.append(f"P_max_{i}")
                i += 1
        elif field_key == "psi_sq":
            i = 0
            while f"scalars/psi_sq_max_w{i}" in h5:
                per_laser_keys.append(f"psi_sq_max_w{i}")
                i += 1

        snapshots = [_read_field_frame(h5, spec, pi) for pi in snap_physical]

        scalar_data = None
        if has_scalar:
            scalar_data = h5[f"scalars/{scalar_key}"][:][sort_order]

        laser_data_map = {}
        for lk in per_laser_keys:
            laser_data_map[lk] = h5[f"scalars/{lk}"][:][sort_order]

    n_cols = len(snap_logical)
    show_scalar_row = has_scalar or per_laser_keys
    n_rows = 2 if show_scalar_row else 1
    height_ratios = [3, 1] if n_rows == 2 else [1]

    fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=height_ratios)

    vmin = min(s.min() for s in snapshots)
    vmax = max(s.max() for s in snapshots)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = _make_norm(spec, vmin, vmax)

    for col, (li, snapshot) in enumerate(zip(snap_logical, snapshots)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(
            snapshot,
            origin="lower",
            extent=extent,
            cmap=spec["cmap"],
            norm=norm,
            **({"vmin": vmin, "vmax": vmax} if norm is None else {}),
            aspect="equal",
        )
        ax.set_title(f"t = {time_sorted[li]:.1f} ps", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"y ($\mu$m)")
        ax.set_xlabel(r"x ($\mu$m)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_scalar_row:
        ax_sc = fig.add_subplot(gs[1, :])

        if has_scalar and scalar_data is not None:
            label = "Total" if per_laser_keys else scalar_key
            ax_sc.plot(
                time_sorted[: len(scalar_data)],
                scalar_data,
                linewidth=1.2,
                color="black",
                label=label,
            )

        for lk in per_laser_keys:
            ld = laser_data_map[lk]
            if lk.startswith("P_max_"):
                laser_label = lk.replace("P_max_", "Laser ")
            else:
                laser_label = lk.replace("psi_sq_max_w", "Spot ")
            ax_sc.plot(
                time_sorted[: len(ld)],
                ld,
                linewidth=0.9,
                alpha=0.8,
                label=laser_label,
            )

        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(scalar_key if has_scalar else "P_max")
        if per_laser_keys:
            if per_laser_keys[0].startswith("P_max_"):
                title_suffix = "per-laser power"
            else:
                title_suffix = r"per-spot max $|\psi|^2$"
        else:
            title_suffix = scalar_key
        ax_sc.set_title(f"{spec['label']} \u2014 {title_suffix} vs time")
        ax_sc.legend(fontsize=8, loc="best")
        ax_sc.grid(True, alpha=0.3)
        for li in snap_logical:
            ax_sc.axvline(
                time_sorted[li], color="gray", linestyle="--", alpha=0.4, linewidth=0.6
            )

    fig.suptitle(
        f"{routine.upper()} \u2014 {spec['label']}", fontsize=14, fontweight="bold"
    )

    out_path = os.path.join(routine_dir(routine), f"{field_key}.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")


def generate_animation(routine, extent):
    with open_h5(routine) as h5:
        sort_order = load_sorted_order(h5)
        time_sorted = h5["time"][:][sort_order]
        n_total = len(time_sorted)

        if HAS_FFMPEG:
            anim_fps = ANIM_FPS
            target_frames = ANIM_FPS * ANIM_TARGET_SECONDS
        else:
            anim_fps = ANIM_GIF_FPS
            target_frames = min(GIF_MAX_FRAMES, ANIM_GIF_FPS * ANIM_TARGET_SECONDS)
        step = max(1, n_total // target_frames)
        anim_indices = list(range(0, n_total, step))
        anim_physical = [int(sort_order[i]) for i in anim_indices]

        field_keys = list(FIELD_SPECS.keys())
        specs = [FIELD_SPECS[k] for k in field_keys]

        n_samples = min(20, n_total)
        sample_step = max(1, (n_total - 1) // max(1, n_samples - 1))
        sample_indices = list(range(0, n_total, sample_step))[:n_samples]
        sample_physical = [int(sort_order[i]) for i in sample_indices]

        vmins = {}
        vmaxs = {}
        norms = {}
        for k, sp in zip(field_keys, specs):
            vals = [_read_field_frame(h5, sp, pi) for pi in sample_physical]
            vmins[k] = min(v.min() for v in vals)
            vmaxs[k] = max(v.max() for v in vals)
            if vmaxs[k] <= vmins[k]:
                vmaxs[k] = vmins[k] + 1e-12
            norms[k] = _make_norm(sp, vmins[k], vmaxs[k])

        first_frames = {
            k: _read_field_frame(h5, sp, anim_physical[0])
            for k, sp in zip(field_keys, specs)
        }

        print(f"    Pre-loading {len(anim_physical)} frames...")
        preloaded = {}
        for k, sp in zip(field_keys, specs):
            preloaded[k] = np.stack(
                [_read_field_frame(h5, sp, pi) for pi in anim_physical]
            )

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    images = {}

    for ax, k, sp in zip(axes, field_keys, specs):
        kw = {"norm": norms[k]} if norms[k] else {"vmin": vmins[k], "vmax": vmaxs[k]}
        images[k] = ax.imshow(
            first_frames[k],
            origin="lower",
            extent=extent,
            cmap=sp["cmap"],
            aspect="equal",
            **kw,
        )
        ax.set_title(sp["label"], fontsize=11)
        ax.set_xlabel(r"x ($\mu$m)")
        ax.set_ylabel(r"y ($\mu$m)")
        fig.colorbar(images[k], ax=ax, fraction=0.046, pad=0.04)

    title_text = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(frame_num):
        idx = anim_indices[frame_num]
        for k in field_keys:
            images[k].set_data(preloaded[k][frame_num])

        t_val = time_sorted[idx]
        title_text.set_text(f"{routine.upper()} \u2014 t = {t_val:.2f} ps")
        return list(images.values()) + [title_text]

    anim = FuncAnimation(
        fig, update, frames=len(anim_indices), blit=False, interval=1000 // anim_fps
    )

    if HAS_FFMPEG:
        from matplotlib.animation import FFMpegWriter

        ext = "mp4"
        writer = FFMpegWriter(fps=anim_fps, codec="h264", bitrate=3000)
    else:
        from matplotlib.animation import PillowWriter

        ext = "gif"
        writer = PillowWriter(fps=anim_fps)

    out_path = os.path.join(routine_dir(routine), f"dynamics.{ext}")
    print(
        f"    Saving {ext.upper()} ({len(anim_indices)} frames, "
        f"ffmpeg={'yes' if HAS_FFMPEG else 'no'})..."
    )
    anim.save(out_path, writer=writer, dpi=ANIM_DPI)
    plt.close(fig)
    print(f"    {out_path}")


COMPARISON_SCALARS = [
    ("psi_sq_max", r"$|\psi|^2_{max}$"),
    ("nI_max", r"$n_I^{max}$"),
    ("nA_max", r"$n_A^{max}$"),
]


def generate_summary(extent):
    available = []
    for r in ROUTINES:
        if os.path.isfile(os.path.join(DATA_DIR, f"{r}.h5")):
            available.append(r)
    if not available:
        return

    routine_data = {}
    for routine in available:
        with open_h5(routine) as h5:
            so = load_sorted_order(h5)
            time_sorted = h5["time"][:][so]
            psi_spec = FIELD_SPECS["psi_sq"]
            last_physical = int(so[len(time_sorted) - 1])
            last_frame = _read_field_frame(h5, psi_spec, last_physical)

            scalars = {}
            for sc_key, _ in COMPARISON_SCALARS:
                if f"scalars/{sc_key}" in h5:
                    scalars[sc_key] = h5[f"scalars/{sc_key}"][:][so]

            routine_data[routine] = {
                "time": time_sorted,
                "last_frame": last_frame,
                "scalars": scalars,
            }

    n_r = len(available)
    n_sc = len(COMPARISON_SCALARS)
    psi_spec = FIELD_SPECS["psi_sq"]
    fig = plt.figure(
        figsize=(max(6 * n_r, 12), 5 + 4.5 * n_sc), constrained_layout=True
    )
    gs = GridSpec(1 + n_sc, n_r, figure=fig, height_ratios=[2] + [1] * n_sc)

    for col, routine in enumerate(available):
        rd = routine_data[routine]
        ax_img = fig.add_subplot(gs[0, col])
        im = ax_img.imshow(
            rd["last_frame"],
            origin="lower",
            extent=extent,
            cmap=psi_spec["cmap"],
            aspect="equal",
        )
        ax_img.set_title(f"{routine.upper()}\nt = {rd['time'][-1]:.1f} ps", fontsize=11)
        ax_img.set_xlabel(r"x ($\mu$m)")
        if col == 0:
            ax_img.set_ylabel(r"y ($\mu$m)")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    for row, (sc_key, sc_label) in enumerate(COMPARISON_SCALARS, start=1):
        ax_sc = fig.add_subplot(gs[row, :])
        for routine in available:
            rd = routine_data[routine]
            if sc_key in rd["scalars"]:
                scalar = rd["scalars"][sc_key]
                style = ROUTINE_STYLES.get(routine, {})
                ax_sc.plot(rd["time"][: len(scalar)], scalar, label=routine, **style)
        ax_sc.set_xlabel("t (ps)")
        ax_sc.set_ylabel(sc_label)
        ax_sc.set_title(f"{sc_label} \u2014 comparison")
        ax_sc.legend(fontsize=10, framealpha=0.9)
        ax_sc.grid(True, alpha=0.3)

    fig.suptitle("Multi-Pump Comparison Summary", fontsize=14, fontweight="bold")

    out_path = os.path.join(RESULTS_DIR, "summary.png")
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"    {out_path}")


def _init_worker(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir


def _generate_field_png_wrapper(args):
    routine, key, extent = args
    generate_field_png(routine, key, extent)


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = args.data_dir
        print(f"  Data dir: {DATA_DIR}")

    print(f"  ffmpeg: {'found' if HAS_FFMPEG else 'NOT found (GIF fallback)'}")

    extent = load_grid_extent()

    for routine in ROUTINES:
        os.makedirs(routine_dir(routine), exist_ok=True)

    available = []
    for routine in ROUTINES:
        h5_path = os.path.join(DATA_DIR, f"{routine}.h5")
        if os.path.isfile(h5_path):
            available.append(routine)
        else:
            print(f"  Skipping {routine}: {h5_path} not found")

    static_tasks = [
        (routine, key, extent) for routine in available for key in FIELD_SPECS
    ]
    if static_tasks:
        n_workers = min(len(static_tasks), os.cpu_count() or 1)
        print(f"  Generating {len(static_tasks)} static plots ({n_workers} workers)...")
        with Pool(n_workers, initializer=_init_worker, initargs=(DATA_DIR,)) as pool:
            pool.map(_generate_field_png_wrapper, static_tasks)

    for routine in available:
        print(f"  {routine.upper()} \u2014 animation:")
        generate_animation(routine, extent)

    print("  SUMMARY:")
    generate_summary(extent)


if __name__ == "__main__":
    main()
