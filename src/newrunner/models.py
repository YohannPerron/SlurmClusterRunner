"""Typed models used by NewRunner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskMode = Literal["per_gpu", "per_node"]


@dataclass(frozen=True)
class SlurmConfig:
    account: str | None
    partition: str | None
    qos: str | None
    dev_qos: str | None
    max_time_hours: int | float
    require_tag: bool = False
    job_name_prefix: str | None = None
    gres: str | None = None
    constraint: str | None = None
    default_time: str | None = None
    dev_time: str | None = None
    min_time: str | None = None
    use_min_time_default: bool = False
    hint: str | None = None
    signal: str | None = None
    output: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourcesConfig:
    gpu_per_node: int
    cpu_per_gpu: int
    task_mode: TaskMode
    require_full_node_gpu: bool = False
    exclusive: bool = False
    multi_gpu_overrides: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathsConfig:
    remote_launcher_dir: str
    default_project_dir: str
    data_dir: str
    log_dir: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentConfig:
    conda_env: str | None
    shell_init: str | None = None
    exports: dict[str, Any] = field(default_factory=dict)
    pre_run: list[str] = field(default_factory=list)
    wandb: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionConfig:
    name: str
    remote_host: str
    gpu_type: str
    slurm: SlurmConfig
    resources: ResourcesConfig
    paths: PathsConfig
    environment: EnvironmentConfig
    default: bool = False
    project: str | None = None
    modules: dict[str, Any] = field(default_factory=dict)
    default_overrides: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    launcher: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
