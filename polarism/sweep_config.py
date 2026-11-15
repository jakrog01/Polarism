"""Resolve and validate sweep override paths."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import re

import yaml

from polarism.config.simulation_parameters import LaserParameters
from polarism.config.validation import validate_config
from polarism.envelopes import available_envelopes
from polarism.profiles import available_profiles
from polarism.solver.solver_compatibility import check_solver_compatibility


_LASER_ENTRY = re.compile(r"lasers\[(\d+)\]\.(.+)\Z")
_GROUP_SECTIONS = {"grid", "solver", "reservoir", "boundary_condition",
                   "compute_engine", "result", "checkpoint", "stop", "sweep", "terms"}


def _mapping_with_type(mapping, value, name):
    registry = available_profiles if name == "profile" else available_envelopes
    factory = registry.lookup(value)
    allowed = set(factory.params)
    if name == "profile":
        allowed.update({"amplitude", "x0", "y0"})
    return {**{key: item for key, item in mapping.items() if key in allowed}, "type": value}


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
            return "group" if field == "laser_type" or field in {"profile.type", "envelope.type"} else "batch"
    if section == "physics":
        from polarism.config.simulation_parameters import PhysicsConstants
        if field in {item.name for item in fields(PhysicsConstants)}:
            return "group" if field in {"dispersion", "spin"} else "batch"
    if section == "noise":
        from polarism.config.simulation_parameters import NoiseParameters
        if field in {item.name for item in fields(NoiseParameters)}:
            return "group" if field in {"enabled", "stochastic"} else "batch"
    if section == "initial_condition":
        from polarism.config.simulation_parameters import InitialConditionParameters
        if field in {item.name for item in fields(InitialConditionParameters)}:
            return "group" if field in {"type", "path"} else "batch"
        if field.startswith("extra.") and len(field) > 6:
            return "batch"
    if section in {"laser", "potential", "drive"}:
        from polarism.config.simulation_parameters import DriveParameters, PotentialParameters
        known = {"laser": LaserParameters, "potential": PotentialParameters, "drive": DriveParameters}[section]
        mappings = {"profile", "envelope"}
        if any(field.startswith(f"{mapping}.") and len(field) > len(mapping) + 1 for mapping in mappings):
            return "group" if field.endswith(".type") else "batch"
        if field in {item.name for item in fields(known)} or section == "laser" and field.startswith("extra.") and len(field) > 6:
            return "group" if field in {"laser_type", "potential_type", "enabled"} else "batch"
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
            elif field.startswith(("profile.", "envelope.")):
                mapping, key = field.split(".", 1)
                if key not in entries[index].get(mapping, getattr(result.laser, mapping)):
                    raise ValueError(f"Unknown sweep path {path!r}; mapping key is absent")
            elif field not in entries[index] and field not in vars(result.laser):
                raise ValueError(f"Unknown sweep path {path!r}")
            result.laser.entry_overrides.setdefault(index, {})[field] = value
            continue
        section, field = path.split(".", 1)
        if section == "terms":
            name, _, key = field.partition(".")
            if key not in result.terms.get(name, {}):
                raise ValueError(f"Unknown sweep path {path!r}; term parameter is absent")
            result.terms[name][key] = value
            continue
        target = getattr(result, section)
        if field.startswith("extra."):
            key = field[6:]
            if key not in target.extra:
                raise ValueError(f"Unknown sweep path {path!r}; extra key is absent")
            target.extra[key] = value
        elif field.startswith(("profile.", "envelope.")):
            mapping, key = field.split(".", 1)
            if not hasattr(target, mapping) or key not in getattr(target, mapping):
                raise ValueError(f"Unknown sweep path {path!r}; mapping key is absent")
            if key == "type":
                setattr(target, mapping, _mapping_with_type(getattr(target, mapping), value, mapping))
            else:
                getattr(target, mapping)[key] = value
        elif hasattr(target, field):
            setattr(target, field, value)
        else:
            raise ValueError(f"Unknown sweep path {path!r}")
    validate_config(result)
    check_solver_compatibility(result)
    return result
