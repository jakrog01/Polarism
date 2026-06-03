# Contributing

Contributions are easiest to review when physics logic, runtime orchestration, and documentation stay cleanly separated.

## Repository conventions

- Keep reusable simulation logic in `polarism/`.
- Keep Slurm and campaign orchestration in `src/polariton_hpc_pipeline/`.
- Do not mix physics kernels with scheduler-specific boilerplate unless the boundary is explicit.

## Code quality expectations

- Add or update type hints on public code.
- Keep docstrings accurate enough for the generated API reference.
- Prefer extending factories and registries over hard-coding new branches into unrelated modules.
- Validate numerical changes with at least the default pytest suite, and use the compliance suite when solver behavior is affected.

## Documentation workflow

Preview the site locally with:

```bash
mkdocs serve
```

Before opening a pull request:

```bash
mkdocs build
pytest
```

## Pull request checklist

- The implementation is covered by tests or justified as documentation-only.
- Public symbols have updated docstrings and type hints.
- New options are documented in the user guide or API reference.
- Cluster-specific behavior remains separate from the physics simulation kernel.
