"""Partition configuration loading."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.models import (
    EnvironmentConfig,
    PartitionConfig,
    PathsConfig,
    ResourcesConfig,
    SlurmConfig,
)


class ConfigError(ValueError):
    """Raised when partition configuration is invalid."""


CONTROL_PARTITION_KEYS = ("PARTITION", "GPU_TYPE")


def default_partitions_dir() -> Path:
    """Return the partition directory shipped with this runner checkout.

    Console scripts may be invoked from any current working directory, so the
    built-in partition files must be resolved relative to the installed runner
    code instead of relative to ``Path.cwd()``.
    """

    return Path(__file__).resolve().parent.parent / "partitions"


def load_partitions(partitions_dir: Path | str | None = None) -> list[PartitionConfig]:
    """Load and validate all ``*.yaml`` partition files from a directory."""

    directory = default_partitions_dir() if partitions_dir is None else Path(partitions_dir)
    if not directory.exists():
        raise ConfigError(f"Partition directory does not exist: {directory}")

    partitions = [load_partition_file(path) for path in sorted(directory.glob("*.yaml"))]
    if not partitions:
        raise ConfigError(f"No partition YAML files found in {directory}")
    return partitions


def load_partition_file(path: Path | str) -> PartitionConfig:
    """Load one partition YAML file."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{file_path}: expected a YAML mapping at top level")
    return _partition_from_mapping(data, source_path=str(file_path))


def select_partition(
    partitions: list[PartitionConfig],
    partition_name: str | None = None,
    gpu_type: str | None = None,
) -> PartitionConfig:
    """Select a partition by name or by the single configured default.

    ``gpu_type`` is a deprecated compatibility alias for ``partition_name``.
    """

    if partition_name and gpu_type:
        raise ConfigError("Use PARTITION, not both PARTITION and deprecated GPU_TYPE")
    if gpu_type:
        warnings.warn(
            "GPU_TYPE is deprecated; use PARTITION instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        partition_name = gpu_type

    if partition_name:
        for partition in partitions:
            if partition.name == partition_name:
                return partition
        known = ", ".join(partition.name for partition in partitions)
        raise ConfigError(f"Unknown partition '{partition_name}'. Available: {known}")

    defaults = [partition for partition in partitions if partition.default]
    if len(defaults) == 1:
        return defaults[0]
    if not defaults:
        raise ConfigError("No PARTITION provided and no default partition configured")
    names = ", ".join(partition.name for partition in defaults)
    raise ConfigError(f"Multiple default partitions configured: {names}")


def load_selected_partition(
    partitions_dir: Path | str | None = None,
    partition_name: str | None = None,
    gpu_type: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> PartitionConfig:
    """Load all partitions and select one using args or environment variables."""

    env = os.environ if environ is None else environ
    partition_name = partition_name or env.get("PARTITION")
    gpu_type = gpu_type or env.get("GPU_TYPE")
    return select_partition(load_partitions(partitions_dir), partition_name, gpu_type)


def _partition_from_mapping(data: Mapping[str, Any], source_path: str) -> PartitionConfig:
    required_top = ("name", "remote_host", "gpu_type", "slurm", "resources", "paths", "environment")
    for key in required_top:
        _require(data, key, source_path)

    slurm = _slurm(_mapping(data["slurm"], "slurm", source_path), source_path)
    resources = _resources(_mapping(data["resources"], "resources", source_path), source_path)
    paths = _paths(_mapping(data["paths"], "paths", source_path), source_path)
    environment = _environment(_mapping(data["environment"], "environment", source_path), source_path)

    known = set(required_top) | {
        "default",
        "project",
        "modules",
        "default_overrides",
        "features",
        "launcher",
    }
    extra = {key: value for key, value in data.items() if key not in known}
    return PartitionConfig(
        name=_string(data["name"], "name", source_path),
        remote_host=_string(data["remote_host"], "remote_host", source_path),
        gpu_type=_string(data["gpu_type"], "gpu_type", source_path),
        default=bool(data.get("default", False)),
        slurm=slurm,
        resources=resources,
        paths=paths,
        environment=environment,
        project=data.get("project"),
        modules=dict(data.get("modules") or {}),
        default_overrides=dict(data.get("default_overrides") or {}),
        features=dict(data.get("features") or {}),
        launcher=dict(data.get("launcher") or {}),
        source_path=source_path,
        extra=extra,
    )


def _slurm(data: Mapping[str, Any], source_path: str) -> SlurmConfig:
    for key in ("account", "partition", "qos", "dev_qos", "max_time_hours", "require_tag"):
        _require(data, key, f"{source_path}:slurm")
    known = set(SlurmConfig.__dataclass_fields__)
    return SlurmConfig(
        account=data.get("account"),
        partition=data.get("partition"),
        qos=data.get("qos"),
        dev_qos=data.get("dev_qos"),
        max_time_hours=data["max_time_hours"],
        require_tag=bool(data.get("require_tag", False)),
        job_name_prefix=data.get("job_name_prefix"),
        gres=data.get("gres"),
        constraint=data.get("constraint"),
        default_time=str(data["default_time"]) if data.get("default_time") is not None else None,
        dev_time=str(data["dev_time"]) if data.get("dev_time") is not None else None,
        min_time=str(data["min_time"]) if data.get("min_time") is not None else None,
        use_min_time_default=bool(data.get("use_min_time_default", False)),
        hint=data.get("hint"),
        signal=data.get("signal"),
        output=data.get("output"),
        error=data.get("error"),
        extra={key: value for key, value in data.items() if key not in known},
    )


def _resources(data: Mapping[str, Any], source_path: str) -> ResourcesConfig:
    for key in ("gpu_per_node", "cpu_per_gpu", "task_mode"):
        _require(data, key, f"{source_path}:resources")
    task_mode = data["task_mode"]
    if task_mode not in ("per_gpu", "per_node"):
        raise ConfigError(f"{source_path}: resources.task_mode must be 'per_gpu' or 'per_node'")
    known = set(ResourcesConfig.__dataclass_fields__)
    return ResourcesConfig(
        gpu_per_node=int(data["gpu_per_node"]),
        cpu_per_gpu=int(data["cpu_per_gpu"]),
        task_mode=task_mode,
        require_full_node_gpu=bool(data.get("require_full_node_gpu", False)),
        exclusive=bool(data.get("exclusive", False)),
        multi_gpu_overrides=dict(data.get("multi_gpu_overrides") or {}),
        extra={key: value for key, value in data.items() if key not in known},
    )


def _paths(data: Mapping[str, Any], source_path: str) -> PathsConfig:
    for key in ("remote_launcher_dir", "default_project_dir", "data_dir", "log_dir"):
        _require(data, key, f"{source_path}:paths")
    known = set(PathsConfig.__dataclass_fields__)
    return PathsConfig(
        remote_launcher_dir=_string(data["remote_launcher_dir"], "paths.remote_launcher_dir", source_path),
        default_project_dir=_string(data["default_project_dir"], "paths.default_project_dir", source_path),
        data_dir=_string(data["data_dir"], "paths.data_dir", source_path),
        log_dir=_string(data["log_dir"], "paths.log_dir", source_path),
        extra={key: value for key, value in data.items() if key not in known},
    )


def _environment(data: Mapping[str, Any], source_path: str) -> EnvironmentConfig:
    for key in ("exports",):
        _require(data, key, f"{source_path}:environment")
    known = set(EnvironmentConfig.__dataclass_fields__)
    return EnvironmentConfig(
        activate_command=data.get("activate_command"),
        shell_init=data.get("shell_init"),
        exports=dict(data.get("exports") or {}),
        pre_run=list(data.get("pre_run") or []),
        wandb=dict(data.get("wandb") or {}),
        extra={key: value for key, value in data.items() if key not in known},
    )


def _require(data: Mapping[str, Any], key: str, context: str) -> None:
    if key not in data:
        raise ConfigError(f"{context}: missing required field '{key}'")


def _mapping(value: Any, key: str, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context}: field '{key}' must be a mapping")
    return value


def _string(value: Any, key: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{context}: field '{key}' must be a non-empty string")
    return value
