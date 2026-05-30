# Polarism

Polarism is a **Gross-Pitaevskii Equation solver** oriented toward fast 2-D driven-dissipative simulations, GPU acceleration, and reproducible batch execution on HPC systems. This site uses a docs-as-code setup built around MkDocs Material and GitHub Pages deployment.

## What is documented here

- A fast path for local installation and first simulation runs.
- User-facing configuration guidance for grids, pumps, solvers, and outputs.
- Example workflows built on top of `polarism`, including the Slurm-oriented `src/` pipeline.
- Curated API reference focused on public abstractions, module roles, and supported options.
- Clear separation between the reusable `polarism` package and the optional example workflows in `src/`.
- Development notes for physics background, testing, CI, and contribution workflow.

## Architecture at a glance

The repository is split along clean responsibility boundaries:

- `polarism/` contains the reusable simulation library: configuration dataclasses, solver implementations, result handling, and the `SimulationController` class.
- `src/` contains example HPC-oriented workflows built on top of `polarism`, including `pump_multi_comparison`, `threshold_finder`, `create_characteristic`, and `dot_response_fit`. They are not the core package API and are best read as reference workflows.
- `tests/` contains correctness, compliance, and benchmark-style validation suites.
- `docs/` and `mkdocs.yml` define the documentation site and generated API reference.

## Core capabilities

- CPU and optional CuPy-backed GPU execution through a shared compute engine abstraction.
- Multiple solver families, including finite-difference RK4, fused CUDA RK4, split-step FFT, ETD-RK2, and interaction-picture RK4.
- Configurable grid topology, boundary absorption, pumps, reservoirs, and result storage.
- Batch-oriented HDF5, JSON, and NumPy output paths plus real-time visualization hooks.
- Slurm pipeline support for multi-scenario parameter studies, scalar threshold scans, characteristic maps, and response-fit campaigns on cluster hardware.

## Documentation pillars

### Getting Started

Use this section when you want the shortest path to a working environment and a first 2-D run.
Start with [Getting Started](getting-started/index.md), then continue to [Installation](getting-started/installation.md) or [Quickstart](getting-started/quickstart.md).

### User Guide

Use this section when you need to understand how the configuration maps onto the physics model and solver behavior.

### API Reference

Use this section for an abstract view of the package: module responsibilities, interfaces, factories, and supported options, without exposing source-level implementation details.

### Examples

Use this section when you want documented reference workflows that use `polarism` as a library rather than extending the package itself.

### Development and Theory

Use this section when you need the mathematical model, the test strategy, or the repository conventions for contributing code.

## Acknowledgements

This project is co-developed with the [Exciton-Polariton research group](https://polariton.fuw.edu.pl/) working on the Faculty of Physics at University of Warsaw, and is supported by the computing infrastructure of [ICM](https://icm.edu.pl/) University of Warsaw.
