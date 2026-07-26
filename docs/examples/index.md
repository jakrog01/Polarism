# Examples

This section documents the example applications that live in `src/`. These are not part of the public `polarism` package API. They are reference workflows showing how to assemble the package into larger studies, batch pipelines, and cluster jobs.

## What belongs here

- reusable example applications built on top of `polarism`
- cluster or batch workflows that use the package as an engine
- reference project layouts that show how to structure nontrivial studies

## Current examples

- [Example Results](example-results.md): short movies from a five-spot network showing suppressed and activated central condensation
- [polariton_hpc_pipeline](polariton-hpc-pipeline.md): a Slurm-oriented pipeline for threshold search, scenario sweeps, per-scenario visualization, and final aggregation
- [threshold_finder](threshold-finder.md): a scalar GPU sweep for estimating a condensation threshold or scanning pulse separation without retaining field data
- [create_characteristic](create-characteristic.md): a two-dimensional map of peak condensate density over pulse energy and pulse separation
- [dot_response_fit](dot-response-fit.md): a Slurm-oriented MNIST batch workflow for fitting the Gaussian pump spot size against time-only ODE reference traces
- [mnist_digits_polariton_snn_dynamic](mnist-digits-polariton-snn-dynamic.md): a dynamic MNIST SNN workflow with a CPU-side pitch/sigma discretization gate and optional pump-allocation profiling

## Relationship to the package

Treat `polarism/` as the stable simulation library and `src/` as a place for executable examples. If an idea becomes part of the general-purpose simulation API, it should move into `polarism/`. If it remains a campaign-specific orchestration layer, it belongs in `src/`.
