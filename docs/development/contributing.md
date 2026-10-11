# Contributing

Contributions are easiest to review when physics logic, runtime orchestration, and documentation stay cleanly separated.

## Repository conventions

- Keep reusable simulation logic in `polarism/`.
- Do not mix physics kernels with scheduler-specific boilerplate unless the boundary is explicit.

## Code quality expectations

- Add or update type hints on public code.
- Keep docstrings accurate for the generated Python API reference. Update the
  hand-written pages in `docs/api/` when options or behavior change.
- Prefer extending factories and registries over hard-coding new branches into unrelated modules.
- Validate numerical changes with at least the default pytest suite, and use the compliance suite when solver behavior is affected.

## Branches and releases

- `master` holds released code only. Every merge into `master` is a release.
- `dev` is the integration branch. All work lands in `dev` first, through
  short-lived feature branches or direct commits.
- Pull requests into `master` must come from `dev`; the release workflow
  rejects any other source branch.

To release:

1. On `dev`, set the new version in both `pyproject.toml` and
   `polarism/__init__.py` (`__version__`). The two values must match.
   Also update `version` and `date-released` in `CITATION.cff`.
2. Open a pull request from `dev` into `master`. The release workflow fails if
   the versions differ or if the tag `v<version>` already exists.
3. Merge the pull request. The release workflow runs tests, builds the sdist
   and wheel, publishes distributions to PyPI through Trusted Publishing,
   creates the tag `v<version>` with a GitHub release, and deploys the
   documentation site to GitHub Pages. The `pypi` environment and repository
   workflow must be configured as a trusted publisher on PyPI.

## Documentation workflow

Create an environment and install the development extra:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Run the default CPU-oriented tests with `.venv/bin/pytest -q`. On CUDA hardware,
use `.venv/bin/pytest -q --use-gpu` for GPU-marked tests. The
[Testing and CI](testing-and-ci.md) page has the complete CPU/GPU command.

Preview the site locally with:

```bash
.venv/bin/mkdocs serve
```

Before opening a pull request:

```bash
.venv/bin/mkdocs build --strict
.venv/bin/pytest -q
```

## Pull request checklist

- Pull requests target `dev`; only release pull requests target `master`.

- The implementation is covered by tests or justified as documentation-only.
- Public symbols have updated docstrings and type hints.
- New options are documented in the user guide or API reference.
- Cluster-specific behavior remains separate from the physics simulation kernel.
