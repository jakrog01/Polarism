# Contributing

Contributions are easiest to review when physics logic, runtime orchestration, and documentation stay cleanly separated.

## Repository conventions

- Keep reusable simulation logic in `polarism/`.
- Do not mix physics kernels with scheduler-specific boilerplate unless the boundary is explicit.

## Code quality expectations

- Add or update type hints on public code.
- Keep docstrings accurate for library users and maintain the corresponding
  hand-written API reference pages in `docs/api/`.
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
2. Open a pull request from `dev` into `master`. The release workflow fails if
   the versions differ or if the tag `v<version>` already exists.
3. Merge the pull request. The release workflow then runs the tests, builds the
   sdist and wheel, creates the tag `v<version>` with a GitHub release
   containing the distributions and generated notes, and deploys the
   documentation site to GitHub Pages.

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

- Pull requests target `dev`; only release pull requests target `master`.

- The implementation is covered by tests or justified as documentation-only.
- Public symbols have updated docstrings and type hints.
- New options are documented in the user guide or API reference.
- Cluster-specific behavior remains separate from the physics simulation kernel.
