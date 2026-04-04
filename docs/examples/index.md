# Examples

This section documents the example applications that live in `src/`. These are not part of the public `polarism` package API. They are reference workflows showing how to assemble the package into larger studies, batch pipelines, and cluster jobs.

## What belongs here

- reusable example applications built on top of `polarism`
- cluster or batch workflows that use the package as an engine
- reference project layouts that show how to structure nontrivial studies

## Current examples

- [pump_multi_comparison](pump-multi-comparison.md): a Slurm-oriented pipeline for threshold search, scenario sweeps, per-scenario visualization, and final aggregation

## Relationship to the package

Treat `polarism/` as the stable simulation library and `src/` as a place for executable examples. If an idea becomes part of the general-purpose simulation API, it should move into `polarism/`. If it remains a campaign-specific orchestration layer, it belongs in `src/`.
