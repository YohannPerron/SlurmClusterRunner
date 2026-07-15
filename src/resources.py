"""SLURM resource calculation."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import PartitionConfig


class ResourceError(ValueError):
    """Raised when requested resources are invalid for a partition."""


@dataclass(frozen=True)
class ResourceRequest:
    """Computed SLURM resource request for one job."""

    nodes: int
    gpus_per_node: int
    ntasks_per_node: int
    cpus_per_task: int
    total_gpus: int


def calculate_resources(
    total_gpus: int,
    partition: PartitionConfig,
    total_cpus: int | None = None,
) -> ResourceRequest:
    """Compute node/task layout for the requested resources on ``partition``."""

    if total_gpus < 1:
        raise ResourceError("GPU must be >= 1")

    cfg = partition.resources
    if cfg.cpu_per_gpu < 1:
        raise ResourceError("partition resources.cpu_per_gpu must be >= 1")
    if cfg.gpu_per_node < 0:
        raise ResourceError("partition resources.gpu_per_node must be >= 0")

    if cfg.gpu_per_node == 0:
        if total_gpus != 1:
            raise ResourceError(f"partition '{partition.name}' is CPU-only; use GPU=1")
        if total_cpus is not None and total_cpus < 1:
            raise ResourceError("CPU must be >= 1")
        return ResourceRequest(
            nodes=1,
            gpus_per_node=0,
            ntasks_per_node=1,
            cpus_per_task=total_cpus or cfg.cpu_per_node or cfg.cpu_per_gpu,
            total_gpus=0,
        )

    if total_cpus is not None:
        raise ResourceError("CPU can only be used with a CPU-only partition")

    if total_gpus <= cfg.gpu_per_node:
        nodes = 1
        if cfg.require_full_node_gpu:
            gpus_per_node = cfg.gpu_per_node
            if total_gpus != cfg.gpu_per_node:
                raise ResourceError(
                    f"partition '{partition.name}' requires full-node GPU requests "
                    f"({cfg.gpu_per_node} GPUs)"
                )
        else:
            gpus_per_node = total_gpus
    else:
        if total_gpus % cfg.gpu_per_node != 0:
            raise ResourceError(
                f"GPU={total_gpus} must be divisible by gpu_per_node="
                f"{cfg.gpu_per_node} for multi-node jobs"
            )
        nodes = total_gpus // cfg.gpu_per_node
        gpus_per_node = cfg.gpu_per_node

    if cfg.task_mode == "per_gpu":
        ntasks_per_node = gpus_per_node
        cpus_per_task = cfg.cpu_per_gpu
    elif cfg.task_mode == "per_node":
        ntasks_per_node = 1
        cpus_per_task = cfg.cpu_per_gpu * gpus_per_node
    else:
        raise ResourceError("resources.task_mode must be 'per_gpu' or 'per_node'")

    return ResourceRequest(
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
        total_gpus=total_gpus,
    )
