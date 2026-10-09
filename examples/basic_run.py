"""Minimal Polarism run: a continuous Gaussian pump on a periodic CPU grid."""
import polarism as ps
from polarism.compute_engine import compute_engine


def main() -> None:
    cfg = ps.Config()

    cfg.grid.nx = 128
    cfg.grid.ny = 128
    cfg.grid.lx = 100.0
    cfg.grid.ly = 100.0
    cfg.grid.grid_type = "periodic"

    cfg.reservoir.reservoir_type = "single"
    cfg.potential.potential_type = "zero"

    cfg.laser.mode = "single"
    cfg.laser.laser_type = "continuous-gaussian"
    cfg.laser.P0 = 0.02
    cfg.laser.sigma_space = 12.0

    cfg.solver.method = "rk4-fdm"
    cfg.solver.dt = 1e-3
    cfg.solver.total_time = 2.0

    cfg.result.save_results = True
    cfg.result.save_hdf5 = True
    cfg.result.output_directory = "quickstart_results"

    cfg.compute_engine.use_gpu = False
    compute_engine.configure(cfg.compute_engine)

    ps.SimulationController(cfg).run()


if __name__ == "__main__":
    main()
