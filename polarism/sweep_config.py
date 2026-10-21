"""Resolve and validate sweep override paths."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import re

import yaml

from polarism.config.simulation_parameters import LaserParameters
from polarism.config.validation import validate_config
from polarism.solver.solver_compatibility import check_solver_compatibility


_LASER_ENTRY = re.compile(r"lasers\[(\d+)\]\.(.+)\Z")
_GROUP_SECTIONS = {"grid", "solver", "reservoir", "boundary_condition",
                   "compute_engine", "result", "checkpoint", "sweep"}


def override_kind(path):
    """Return whether a known config path can vary within a batch."""
    match = _LASER_ENTRY.fullmatch(path)
    if match:
        section, field = "lasers", match.group(2)
    else:
        parts = path.split(".")
        section, field = parts[0], ".".join(parts[1:])
    if section == "lasers":
        if field:
            return "batch"
    if section == "physics":
        from polarism.config.simulation_parameters import PhysicsConstants
        if field in {item.name for item in fields(PhysicsConstants)}:
            return "batch"
    if section in {"laser", "potential"}:
        from polarism.config.simulation_parameters import PotentialParameters
        known = LaserParameters if section == "laser" else PotentialParameters
        if field in {item.name for item in fields(known)} or section == "laser" and field.startswith("extra.") and len(field) > 6:
            return "batch"
    if section in _GROUP_SECTIONS and field:
        return "group"
    raise ValueError(f"Unknown sweep path {path!r}; use paths such as physics.g_C, laser.P0, laser.extra.name, potential.V1 or lasers[1].delay")


def apply_overrides(cfg, overrides, *, allow_groups=False):
    """Copy a config and apply dotted paths, preserving the input config."""
    result = deepcopy(cfg)
    for path, value in overrides.items():
        kind = override_kind(path)
        if kind == "group" and not allow_groups:
            raise ValueError(f"{path} is non-batchable; use Sweep to group members")
        match = _LASER_ENTRY.fullmatch(path)
        if match:
            if result.laser.mode != "multiple":
                raise ValueError(f"{path} requires laser.mode='multiple'")
            with open(result.laser.config_file, encoding="utf-8") as handle:
                entries = (yaml.safe_load(handle) or {}).get("lasers", [])
            index, field = int(match.group(1)), match.group(2)
            if index >= len(entries):
                raise ValueError(f"Unknown sweep path {path!r}; laser index is out of range")
            if field.startswith("extra."):
                key = field[6:]
                if key not in entries[index] and key not in entries[index].get("extra", {}) and key not in result.laser.extra:
                    raise ValueError(f"Unknown sweep path {path!r}; extra key is absent")
            elif field not in entries[index] and field not in vars(result.laser):
                raise ValueError(f"Unknown sweep path {path!r}")
            changes = getattr(result, "_laser_entry_overrides", {})
            changes.setdefault(index, {})[field] = value
            result._laser_entry_overrides = changes
            continue
        section, field = path.split(".", 1)
        target = getattr(result, section)
        if field.startswith("extra."):
            key = field[6:]
            if key not in target.extra:
                raise ValueError(f"Unknown sweep path {path!r}; extra key is absent")
            target.extra[key] = value
        elif hasattr(target, field):
            setattr(target, field, value)
        else:
            raise ValueError(f"Unknown sweep path {path!r}")
    validate_config(result)
    check_solver_compatibility(result)
    return result
