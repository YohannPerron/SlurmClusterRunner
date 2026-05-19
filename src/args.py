"""CLI argument and control-parameter parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import Iterable

from src.models import PartitionConfig


CONTROL_PARAMS = {
    "NAME",
    "GPU",
    "PARTITION",
    "GPU_TYPE",
    "BATCH",
    "TIME",
    "MINTIME",
    "DEV",
    "CONDA_ENV",
    "TAG",
}


class ArgumentError(ValueError):
    """Raised when CLI/control arguments are invalid."""


@dataclass(frozen=True)
class RawCliArgs:
    """Top-level parsed CLI arguments before control extraction."""

    executable: str
    tokens: list[str]
    allow_control_sweeps: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class ControlParams:
    """Launcher control parameters consumed before forwarding script args."""

    name: str | None = None
    gpu: int = 1
    partition: str | None = None
    gpu_type: str | None = None
    batch: int | None = None
    time: str | None = None
    mintime: bool | None = None
    dev: bool = False
    conda_env: str | None = None
    tag: str | None = None


@dataclass(frozen=True)
class ControlSweepPlan:
    """Expanded launcher control parameters.

    ``variable_params`` contains control axes with more than one value so later
    path planning can include values such as ``GPU=4`` or ``BATCH=128`` in run
    names.
    """

    controls: list[ControlParams]
    variable_params: list[dict[str, str]]


def parse_cli(argv: Iterable[str]) -> RawCliArgs:
    """Parse the stable CLI shape.

    The remaining tokens are intentionally not interpreted by argparse because
    they can be positional script arguments or Hydra overrides.
    """

    parser = argparse.ArgumentParser(prog="runner")
    parser.add_argument(
        "--allow-control-sweeps",
        action="store_true",
        help="Allow sweeping control parameters other than GPU, PARTITION, and BATCH.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print external commands and timing information.",
    )
    parser.add_argument("executable")
    parser.add_argument("tokens", nargs="*")
    ns = parser.parse_args(list(argv))
    return RawCliArgs(
        ns.executable,
        list(ns.tokens),
        allow_control_sweeps=ns.allow_control_sweeps,
        verbose=ns.verbose,
    )


def split_control_params(tokens: Iterable[str]) -> tuple[ControlParams, list[str]]:
    """Extract launcher control params and return remaining forwarded tokens.

    This compatibility helper requires control parameters to have a single value.
    Use :func:`split_control_param_sweep` when sweep expansion is desired.
    """

    plan, forwarded = split_control_param_sweep(tokens)
    if len(plan.controls) != 1:
        raise ArgumentError("Control parameter sweeps require split_control_param_sweep")
    return plan.controls[0], forwarded


def split_control_param_sweep(tokens: Iterable[str]) -> tuple[ControlSweepPlan, list[str]]:
    """Extract and expand launcher control parameter sweeps.

    Examples: ``GPU=2,4`` expands to two ``ControlParams`` objects with
    ``gpu == 2`` and ``gpu == 4``; ``BATCH=64,128`` behaves similarly.
    Non-control tokens are returned unchanged for script/Hydra sweep parsing.
    """

    values: dict[str, str] = {}
    forwarded: list[str] = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key in CONTROL_PARAMS:
            values[key] = value
        else:
            forwarded.append(token)

    axes = {key: _split_bracket_aware(value) for key, value in values.items()}
    for key, axis_values in axes.items():
        if not axis_values:
            raise ArgumentError(f"{key} must have at least one value")

    keys = list(axes)
    combinations = product(*(axes[key] for key in keys)) if keys else [()]
    controls: list[ControlParams] = []
    variable_params: list[dict[str, str]] = []
    variable_keys = {key for key, axis_values in axes.items() if len(axis_values) > 1}

    for combination in combinations:
        selected = dict(zip(keys, combination, strict=True))
        control = _build_controls(selected)
        controls.append(control)
        variable_params.append({key: selected[key] for key in keys if key in variable_keys})

    return ControlSweepPlan(controls=controls, variable_params=variable_params), forwarded


def _build_controls(values: dict[str, str]) -> ControlParams:
    controls = ControlParams(
        name=_optional(values.get("NAME")),
        gpu=_int(values.get("GPU"), "GPU", default=1),
        partition=_optional(values.get("PARTITION")),
        gpu_type=_optional(values.get("GPU_TYPE")),
        batch=_optional_int(values.get("BATCH"), "BATCH"),
        time=_time(values.get("TIME")),
        mintime=_optional_bool(values.get("MINTIME"), "MINTIME"),
        dev=_bool(values.get("DEV"), "DEV", default=False),
        conda_env=_optional(values.get("CONDA_ENV")),
        tag=_optional(values.get("TAG")),
    )
    if controls.partition and controls.gpu_type:
        raise ArgumentError("Use PARTITION, not both PARTITION and deprecated GPU_TYPE")
    return controls


def validate_control_params(controls: ControlParams, partition: PartitionConfig) -> None:
    """Validate controls that depend on the selected partition."""

    if partition.slurm.require_tag and not controls.tag:
        raise ArgumentError(f"TAG is required for partition '{partition.name}'")


def _split_bracket_aware(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    pairs = {"[": "]", "(": ")", "{": "}"}
    for index, char in enumerate(value):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if char == "," and not stack:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _optional(value: str | None) -> str | None:
    return value if value not in (None, "") else None


def _int(value: str | None, key: str, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentError(f"{key} must be an integer") from exc
    if parsed < 1:
        raise ArgumentError(f"{key} must be >= 1")
    return parsed


def _optional_int(value: str | None, key: str) -> int | None:
    if value is None or value == "":
        return None
    return _int(value, key, default=1)


def _bool(value: str | None, key: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ArgumentError(f"{key} must be a boolean")


def _optional_bool(value: str | None, key: str) -> bool | None:
    if value is None or value == "":
        return None
    return _bool(value, key, default=False)


def _time(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if value.isdigit():
        return value
    parts = value.split(":")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        return value
    raise ArgumentError("TIME must be HH:MM:SS or integer hours")
