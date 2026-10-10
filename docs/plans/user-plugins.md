# Plan: user plugins via `polarism_plugins/`

## Goal
A user who installed Polarism with pip can create a `polarism_plugins/` folder in
their project (current working directory), put modules there that use the
existing registry decorators (`register_laser`, `register_potential`,
`register_solver`, `register_absorption`), and those entries become available
in config (`laser_type`, `potential_type`, `solver.method`, `absorption`) both via
the `polarism` CLI and via `SimulationController` from a script.

## Steps
1. New module `polarism/plugins.py` with `load_plugins(root: str | Path | None = None) -> list[str]`:
   - `root` defaults to `Path.cwd()`; if `root / "polarism_plugins"` is not a dir, return `[]`.
   - Prepend `str(root)` to `sys.path` only if not already present.
   - Import package `polarism_plugins` (works with or without `__init__.py`, i.e.
     namespace packages too, including nested subfolders like `laser/`).
   - Recursively import every module with `pkgutil.walk_packages(pkg.__path__, "polarism_plugins.")`.
   - Return list of imported module names. Idempotent (second call does not re-execute modules).
   - Import errors in a plugin module must propagate (do not swallow), so users see their bugs.
2. Call `load_plugins()` at startup:
   - `polarism/cli.py:main()` before `tyro.cli(...)`.
   - `SimulationController.__init__` before `validate_config(cfg)`.
3. Override warning: in each of the 4 registries, if `name` is already registered
   with a different object, emit `warnings.warn(..., stacklevel=2)` and still overwrite.
4. Custom laser parameters: add `extra: dict[str, Any] = field(default_factory=dict)`
   to `LaserParameters`. In `LaserFactory._create_multiple_lasers`, YAML keys that
   are not dataclass fields go into `extra` (merged on top of any base `extra`), instead of raising `TypeError`.
   `polarism --help` / `tyro.cli(Config, args=[])` must keep working; if tyro cannot
   handle the field type, suppress it from the CLI (`tyro.conf.Suppress`) rather than
   changing its type.
5. Public exports: add `register_potential` to `polarism/potential/__init__.py`,
   `register_solver` to `polarism/solver/__init__.py`, `register_absorption` to
   `polarism/boundary_conditions/absorption/__init__.py` (laser already exports it).
   Export `load_plugins` from `polarism/__init__.py`.
6. Tests: `tests/unit/test_plugins.py` using `tmp_path`:
   - no folder -> `[]`;
   - folder with nested `laser/my_laser.py` registering a laser -> laser is in `available_lasers`
     and `LaserFactory.create_laser` builds it; also a potential plugin;
   - works without `__init__.py` files;
   - multi-laser YAML with unknown key lands in `extra`;
   - duplicate registration warns.
   - Clean up: registry entries, `sys.modules` entries starting with `polarism_plugins`, `sys.path`.
7. Docs: new page `docs/user-guide/plugins.md` (folder layout, laser example
   subclassing `AbstractLaser`, using `extra`, note that plugin code is executed
   from CWD), add to `mkdocs.yml` nav under User Guide.

## Out of scope
No renaming of `laser_registy.py` / `potential_registy.py`, no entry points, no
validation hooks for plugins, no refactors elsewhere.

## Acceptance criteria
- `python -m pytest -q --no-header -m "not slow and not compliance and not gpu"` passes (baseline: 123 passed).
- New tests cover the cases in step 6.
- `python -c "import tyro; from polarism import Config; tyro.cli(Config, args=[])"` works.
- `mkdocs.yml` nav includes the new page.
