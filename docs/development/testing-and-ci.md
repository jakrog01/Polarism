# Testing and CI

Run the fast CPU selection for ordinary changes and the targeted numerical suites when solver behavior changes. GitHub Actions checks the CPU selection and builds documentation, while GPU and long tests run locally on suitable hardware.

## Local commands

```bash
.venv/bin/pytest -q -m "not slow and not compliance and not gpu"
.venv/bin/mkdocs build --strict
```

`pyproject.toml` registers `slow`, `compliance`, and `gpu` markers. Its default pytest expression excludes `slow` and `compliance`, but does not exclude `gpu`. The explicit command above matches the hosted CPU selection. To select a suite explicitly, use:

```bash
.venv/bin/pytest -q -m compliance
.venv/bin/pytest -q -m "gpu and not slow" --use-gpu
.venv/bin/pytest -q -m slow --use-gpu
.venv/bin/pytest -q -m '' --use-gpu
```

A command-line `-m` replaces the default marker expression. The final command includes all markers; run it on a machine with a CUDA device and allow enough time for performance studies. Do not run concurrent GPU benchmarks when comparing timing. Tests under `tests/reference/` check analytic decay, spatial uniformity, and reservoir limits; cross-check tests compare solver behavior.

## Hosted workflows

| Workflow | Trigger | Checks |
| --- | --- | --- |
| `tests.yml` | Push, pull request, or reusable call | Python 3.12 and 3.13; `not slow and not compliance and not gpu` |
| `docs.yml` | Pull request or reusable call | Python 3.13; strict MkDocs build; deploys only when called with `deploy` |
| `release.yml` | Pull request to `master` or push to `master` | Version and tag checks, tests, distributions; publication and release only after push |

The GitHub ruleset requires CI on `master`. The release workflow also rejects a release pull request whose source is not `dev`. Hosted tests do not run the compliance, slow, or GPU suites; run the relevant ones before a numerical release. See [Contributing](contributing.md) for the branch and release process.
