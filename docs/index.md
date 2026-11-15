# Polarism

Polarism is a Python package for 1-D and 2-D driven-dissipative Gross-Pitaevskii simulations of exciton-polariton systems. Start with `pip install polarism`, then follow the [quickstart](getting-started/quickstart.md).

## Highlights

- Scalar or spinor (ψ+, ψ−) polaritons with TE–TM splitting, Zeeman splitting and cross-spin interaction. See [Spin and Polarization](user-guide/spin-and-polarization.md).
- Single, double, quadratic-double or no reservoir, shared or per spin component.
- Incoherent pumps built from spatial profiles, lattices and temporal envelopes, a coherent (resonant) drive, and time-dependent potentials.
- Truncated-Wigner noise from a counter-based Philox generator, reproducible from one seed on CPU and GPU.
- Finite-difference RK4, split-step FFT, ETD-RK2, interaction-picture RK4 and higher-order IFRK solvers on NumPy, CuPy or fused CUDA kernels, with CUDA graph chunks.
- Accumulated observables: momentum distribution, g⁽¹⁾, g⁽²⁾ and Welch energy spectra E(k) and E(x). See [Output and Visualization](user-guide/output-and-visualization.md).
- Adaptive time stepping for the embedded IFRK tableaus. See [Time Evolution](user-guide/time-evolution.md).
- Batched parameter sweeps, checkpoint and restart, asynchronous HDF5/NPY/JSON output and a lazy result reader.
- Registries for solvers, terms, dispersions, profiles, envelopes, initial conditions, observables and output formats.

Release notes for each version are on the [GitHub releases page](https://github.com/jakrog01/Polarism/releases).

## What is documented here

- A fast path for local installation and first simulation runs.
- User-facing configuration guidance for grids, pumps, solvers, and outputs.
- Hand-written reference plus generated Python signatures and docstrings for the public API.
- Development notes for physics background, testing, CI, and contribution workflow.

## Architecture at a glance

The repository is split along clean responsibility boundaries:

- `polarism/` contains the reusable simulation library: configuration dataclasses, solver implementations, result handling, and the `SimulationController` class.
- `tests/` contains correctness, compliance, and benchmark-style validation suites.
- `docs/` and `mkdocs.yml` define the documentation site and generated API reference.

## Core capabilities

- CPU and optional CuPy-backed GPU execution through a shared compute engine abstraction.
- Multiple solver families, including finite-difference RK4, fused CUDA RK4, split-step FFT, ETD-RK2, interaction-picture RK4, and IFRK with RK4/DP5/DOP853/Verner tableaus.
- Configurable grid topology, boundary absorption, pumps, reservoirs, and result storage.
- Batch-oriented HDF5, JSON, and NumPy output paths plus real-time visualization hooks.

## Documentation pillars

### Getting Started

Use this section when you want the shortest path to a working environment and a first run.
Start with [Getting Started](getting-started/index.md), then continue to [Installation](getting-started/installation.md) or [Quickstart](getting-started/quickstart.md).

### User Guide

Use this section when you need to understand how the configuration maps onto the physics model and solver behavior.
To add components, start with the [plugin guide](plugins/index.md).

### Reference

Use this section for package responsibilities, interfaces, factories, and supported options. The [Python API](api/python-api.md) is generated from public objects and docstrings.

### Explanation and Development

Use this section when you need the mathematical model, the test strategy, or the repository conventions for contributing code.

## Acknowledgements

This project is co-developed with the [Exciton-Polariton research group](https://polariton.fuw.edu.pl/) working on the Faculty of Physics at University of Warsaw, and is supported by the computing infrastructure of [ICM](https://icm.edu.pl/) University of Warsaw.
