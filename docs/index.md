# Polarism

Polarism is a Python package for 2-D driven-dissipative Gross-Pitaevskii simulations of exciton-polariton systems. Start with `pip install polarism`, then follow the [quickstart](getting-started/quickstart.md).

## In 0.3.x

- PyPI installation includes optional CUDA 12 and CUDA 13 extras.
- `split-step-fft-cuda` joins the CUDA solver options; eligible runs of it and `rk4-cuda` support CUDA graph chunks.
- `polarism-bench` compares per-step and chunked execution on the current GPU.

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
- Multiple solver families, including finite-difference RK4, fused CUDA RK4, split-step FFT, ETD-RK2, and interaction-picture RK4.
- Configurable grid topology, boundary absorption, pumps, reservoirs, and result storage.
- Batch-oriented HDF5, JSON, and NumPy output paths plus real-time visualization hooks.

## Documentation pillars

### Getting Started

Use this section when you want the shortest path to a working environment and a first 2-D run.
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
