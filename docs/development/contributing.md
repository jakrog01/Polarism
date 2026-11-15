# Contributing

Make changes on `dev` or a short-lived branch targeting `dev`. Keep numerical behavior, tests, and the user-facing documentation in sync.

## Set up the repository

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/mkdocs build --strict
```

The `dev` extra includes pytest and MkDocs. For CUDA work, install a matching CuPy package with the CUDA toolkit headers, for example `cupy-cuda12x[ctk]` on CUDA 12. The repository has no `uv.lock`; `uv pip install` can install into an existing environment.

Keep reusable implementation in `polarism/`, add type hints and accurate docstrings to public APIs, and use registries when introducing new components. Update the relevant guide and [Configuration reference](../reference/configuration.md) when behavior or keys change. Test numerical changes with the default suite and the compliance suite. [Testing and CI](testing-and-ci.md) gives the commands.

## Branches and releases

`dev` is the integration branch. `master` contains releases; every merge into `master` starts the release workflow. The GitHub ruleset requires CI to pass on `master`, and the release workflow accepts pull requests into `master` only from `dev`.

Before opening the release pull request, set the same version in `pyproject.toml` and `polarism/__init__.py`, and update `CITATION.cff` version and release date. The release workflow checks that the version is new and that its `v<version>` tag does not exist. It runs tests and checks built distributions on the pull request. After merging, it publishes to PyPI through Trusted Publishing, creates the tag and GitHub release, and deploys MkDocs to GitHub Pages.

For a normal pull request, target `dev`, describe the behavioral change, and include tests or explain why the change is documentation only. Preview docs with `.venv/bin/mkdocs serve`.
