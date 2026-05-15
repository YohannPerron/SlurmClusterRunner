"""CLI argument and control-parameter parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from newrunner.models import PartitionConfig


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

    project_path: str
    executable: str
    tokens: list[str]


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


def parse_cli(argv: Iterable[str]) -> RawCliArgs:
    """Parse the stable CLI shape.

    The remaining tokens are intentionally not interpreted by argparse because
    they can be positional script arguments or Hydra overrides.
    """

    parser = argparse.ArgumentParser(prog="newrunner")
    parser.add_argument("project_path")
    parser.add_argument("executable")
    parser.add_argument("tokens", nargs="*")
    ns = parser.parse_args(list(argv))
    return RawCliArgs(ns.project_path, ns.executable, list(ns.tokens))


def split_control_params(tokens: Iterable[str]) -> tuple[ControlParams, list[str]]:
    """Extract launcher control params and return remaining forwarded tokens."""

    values: dict[str, str] = {}
    forwarded: list[str] = []
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key in CONTROL_PARAMS:
            values[key] = value
        else:
            forwarded.append(token)

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
    return controls, forwarded


def validate_control_params(controls: ControlParams, partition: PartitionConfig) -> None:
    """Validate controls that depend on the selected partition."""

    if partition.slurm.require_tag and not controls.tag:
        raise ArgumentError(f"TAG is required for partition '{partition.name}'")


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
