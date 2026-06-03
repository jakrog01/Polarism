"""Config loading and basic extraction utilities.

Responsible for: reading YAML, resolving power/delay expressions, extracting
named sections from the config dict.  No validation, no dataclass construction.
Those live in validator.py and builder.py.
"""
from __future__ import annotations

import ast
import math
import operator as _op
import re
from dataclasses import fields as dc_fields
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_BINOPS: dict[type, Any] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
}
_UNOPS: dict[type, Any] = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}
_POWER_EXPR_RE = re.compile(
    r"^\s*([0-9]*\.?[0-9]*)\s*"
    r"(P(?:_(?:ring|probe|assisted|assisted_probe|probe_assisted|"
    r"assisted_ring|ring_assisted))?"
    r"|P(?:ring|probe|assisted|assistedprobe|probeassisted|"
    r"assistedring|ringassisted))\s*$",
    re.IGNORECASE,
)


def _eval_arith(node: ast.expr, namespace: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numeric constants allowed, got {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise ValueError(f"Unknown variable {node.id!r}")
        return float(namespace[node.id])
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported operator {type(node.op).__name__!r}")
        left = _eval_arith(node.left, namespace)
        right = _eval_arith(node.right, namespace)
        try:
            return fn(left, right)
        except ZeroDivisionError:
            raise ValueError("Division by zero in delay expression")
    if isinstance(node, ast.UnaryOp):
        fn = _UNOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__!r}")
        return fn(_eval_arith(node.operand, namespace))
    raise ValueError(f"Unsupported expression node {type(node).__name__!r}")


def load_config(path: str) -> dict[str, Any]:
    """Load and return the raw config dict from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_power(
    expr: str | float | None,
    p_threshold: float,
    refs: dict[str, float] | None = None,
) -> float:
    """Resolve a power expression to a float.

    Supports numeric literals and fractional-P notation such as ``"0.6P"``.
    Optional named references ``P_ring``, ``P_probe`` and
    ``P_assisted_ring`` are used by probe5 calibration runs where the ring,
    center-probe, and center-assisted ring thresholds are measured
    independently.  Plain ``P`` remains the legacy threshold alias.

    Parameters
    ----------
    expr : str, float, or None
        Power value or expression.  ``None`` returns *p_threshold* unchanged.
    p_threshold : float
        Reference threshold power.
    refs : dict, optional
        Additional reference powers.  Recognised keys are ``P_ring``,
        ``P_probe`` and ``P_assisted_ring``; ``*_threshold`` names are
        accepted as threshold-result aliases.

    Returns
    -------
    float
        Resolved power in the same units as *p_threshold*.
    """
    if expr is None:
        return p_threshold
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, str):
        m = _POWER_EXPR_RE.match(expr)
        if m:
            coeff_str = m.group(1)
            coeff = float(coeff_str) if coeff_str else 1.0
            symbol = m.group(2).lower().replace("_", "")
            power_refs = refs or {}
            if symbol == "p":
                reference = p_threshold
            elif symbol == "pring":
                reference = power_refs.get(
                    "P_ring",
                    power_refs.get("P_ring_threshold", p_threshold),
                )
            elif symbol == "pprobe":
                reference = power_refs.get(
                    "P_probe",
                    power_refs.get("P_probe_threshold", p_threshold),
                )
            elif symbol in {"passisted", "passistedprobe", "pprobeassisted"}:
                reference = power_refs.get(
                    "P_assisted_probe",
                    power_refs.get(
                        "P_assisted_probe_threshold",
                        power_refs.get("P_probe_assisted_threshold", p_threshold),
                    ),
                )
            elif symbol in {"passistedring", "pringassisted"}:
                reference = power_refs.get(
                    "P_assisted_ring",
                    power_refs.get(
                        "P_assisted_ring_threshold",
                        power_refs.get("P_ring_assisted_threshold", p_threshold),
                    ),
                )
            else:
                raise ValueError(f"Unknown power reference {m.group(2)!r}")
            return coeff * float(reference)
        return float(expr)
    raise ValueError(f"Invalid power expression: {expr!r}")


def resolve_delay(
    expr: str | int | float | None,
    namespace: dict[str, float],
) -> float:
    """Resolve a delay expression to a float (ps).

    Parameters
    ----------
    expr : str, int, float, or None
        Delay value.  ``None`` resolves to ``0.0``.  A ``str`` is parsed as
        an arithmetic expression; allowed names are those in *namespace*.
        Only ``+``, ``-``, ``*``, ``/`` and numeric constants are permitted —
        no builtins, no function calls, no imports.
    namespace : dict[str, float]
        Variables available inside the expression.  Typically built by
        :func:`build_timing_namespace`.

    Returns
    -------
    float
        Resolved delay in ps.

    Raises
    ------
    ValueError
        If *expr* is not a recognised type, contains a syntax error, uses an
        unknown variable, or uses a disallowed AST node.
    """
    if expr is None:
        return 0.0
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, str):
        try:
            tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid delay expression {expr!r}: {exc}") from exc
        return _eval_arith(tree.body, namespace)
    raise ValueError(
        f"delay must be a number or expression string, got {type(expr).__name__!r}"
    )


def build_timing_namespace(
    threshold: dict[str, Any],
    defaults: dict[str, Any],
    timing_vars: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Build the variable namespace for delay expression evaluation.

    The base namespace always contains three threshold-derived quantities:

    * ``sigma_time`` — temporal width of one pulse (ps)
    * ``pulse_separation`` — period between consecutive pulse starts (ps)
    * ``cutoff_sigma`` — pulse cutoff multiplier (dimensionless)

    Scenarios may extend this namespace via a ``timing_vars`` block, naming
    derived quantities in physically readable terms such as ``pulse_duration``
    or ``cycle_duration``.  Each timing variable is itself an expression
    evaluated against the current namespace, so later variables may reference
    earlier ones.

    Parameters
    ----------
    threshold : dict
        Threshold search result (``threshold_result.json``).
    defaults : dict
        Global ``laser_defaults`` section.
    timing_vars : dict, optional
        Scenario-level ``timing_vars`` mapping.  Values may be numbers or
        expression strings using base variables or previously defined vars.

    Returns
    -------
    dict[str, float]
        Combined namespace: base variables plus evaluated timing_vars.
    """
    ns: dict[str, float] = {
        "sigma_time": float(threshold.get("sigma_time", 1.0)),
        "pulse_separation": float(threshold.get("pulse_separation", 10.0)),
        "cutoff_sigma": float(defaults.get("cutoff_sigma", 3.0)),
    }
    for name, expr in (timing_vars or {}).items():
        ns[name] = resolve_delay(expr, dict(ns))
    return ns


def make_dataclass(cls: type[T], overrides: dict[str, Any]) -> T:
    """Construct *cls* from *overrides*, ignoring unknown keys.

    Parameters
    ----------
    cls : type
        A dataclass type.
    overrides : dict
        Key-value pairs; unknown keys (not fields of *cls*) are silently
        ignored so YAML sections can contain extra annotations.

    Returns
    -------
    T
        Instance of *cls*.
    """
    valid = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in overrides.items() if k in valid})


def get_laser_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``laser_defaults`` dict from the ``global`` section."""
    return cfg["global"].get("laser_defaults", {})


def get_threshold_search_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the ``threshold_search`` sub-dict from the ``global`` section."""
    return cfg["global"]["threshold_search"]


def get_scenario(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the scenario entry with the given *name*.

    Raises
    ------
    KeyError
        If no scenario with that name exists.
    """
    for sc in cfg["scenarios"]:
        if sc["name"] == name:
            return sc
    raise KeyError(f"Scenario '{name}' not found in config")
