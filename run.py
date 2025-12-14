import tyro

import polarism as ps


def main():
    cfg = tyro.cli(ps.Config)
    controller = ps.SimulationController(cfg)
    controller.run()


if __name__ == "__main__":
    main()
