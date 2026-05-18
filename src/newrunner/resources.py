"""SLURM resource calculation."""

from __future__ import annotations

from dataclasses import dataclass

from newrunner.models import PartitionConfig


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


def calculate_resources(total_gpus: int, partition: PartitionConfig) -> ResourceRequest:
    """Compute node/task layout for ``total_gpus`` on ``partition``."""

    if total_gpus < 1:
        raise ResourceError("GPU must be >= 1")

    cfg = partition.resources
    if cfg.gpu_per_node < 1:
        raise ResourceError("partition resources.gpu_per_node must be >= 1")
    if cfg.cpu_per_gpu < 1:
        raise ResourceError("partition resources.cpu_per_gpu must be >= 1")

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
