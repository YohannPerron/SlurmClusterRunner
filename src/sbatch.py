"""sbatch script generation."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from src.args import ControlParams
from src.models import PartitionConfig
from src.paths import RunPathPlan
from src.resources import ResourceRequest
from src.sweep import SweepJob
from src.time import TimeRequest, parse_wall_time


@dataclass(frozen=True)
class SbatchContext:
    """Inputs needed to render one sbatch script."""

    partition: PartitionConfig
    project_path: str
    executable: str
    controls: ControlParams
    sweep_job: SweepJob
    resources: ResourceRequest
    time: TimeRequest
    paths: RunPathPlan


def render_sbatch(ctx: SbatchContext) -> str:
    """Render a complete sbatch script for one expanded job."""

    lines = ["#!/bin/bash", * _slurm_header(ctx), "", "set -euo pipefail", ""]
    lines.extend(_environment_lines(ctx))
    lines.append("")
    lines.append(f"cd {shlex.quote(ctx.project_path)}")
    lines.append(_command_line(ctx))
    lines.append("")
    return "\n".join(lines)


def _slurm_header(ctx: SbatchContext) -> list[str]:
    p = ctx.partition
    s = p.slurm
    r = ctx.resources
    job_name_parts = [part for part in [s.job_name_prefix, PurePosixPath(ctx.executable).stem, ctx.controls.name] if part]
    header = [f"#SBATCH --job-name={_slurm_value('-'.join(job_name_parts) or 'newrunner')}"]
    optional = {
        "account": s.account,
        "partition": s.partition,
        "qos": s.dev_qos if ctx.controls.dev and s.dev_qos else s.qos,
        "constraint": s.constraint,
        "gres": f"{s.gres}:{r.gpus_per_node}" if s.gres else None,
        "nodes": r.nodes,
        "gpus-per-node": r.gpus_per_node,
        "ntasks-per-node": r.ntasks_per_node,
        "cpus-per-task": r.cpus_per_task,
        "time": ctx.time.wall_time,
        "time-min": _effective_min_time(ctx),
        "hint": s.hint,
        "signal": s.signal,
        "output": s.output or ctx.paths.stdout_path,
        "error": s.error or ctx.paths.stderr_path,
        "comment": ctx.controls.tag,
    }
    for key, value in optional.items():
        if value is not None:
            header.append(f"#SBATCH --{key}={_slurm_value(value)}")
    if p.resources.exclusive:
        header.append("#SBATCH --exclusive")
    for key, value in s.extra.items():
        header.append(f"#SBATCH --{key}={_slurm_value(value)}")
    return header


def _effective_min_time(ctx: SbatchContext) -> str | None:
    """Return a valid --time-min value, if enabled.

    Slurm requires --time-min to be strictly lower than --time.  Some
    partitions define a long default min_time (for normal jobs) but a much
    shorter dev_time; in that case emitting --time-min would make dev jobs
    invalid, so omit it.
    """

    min_time = ctx.partition.slurm.min_time
    if not ctx.time.use_min_time or not min_time:
        return None
    if parse_wall_time(str(min_time)) >= ctx.time.seconds:
        return None
    return min_time


def _environment_lines(ctx: SbatchContext) -> list[str]:
    p = ctx.partition
    env = p.environment
    lines: list[str] = []
    if p.modules.get("purge"):
        lines.append("module purge")
    for module in p.modules.get("load", []) or []:
        lines.append(f"module load {shlex.quote(str(module))}")
    if env.shell_init:
        lines.append(f"source {shlex.quote(env.shell_init)}")
    conda_env = ctx.controls.conda_env or env.conda_env
    if conda_env:
        lines.append(f"conda activate {shlex.quote(conda_env)}")
    for key, value in env.exports.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.extend(env.pre_run)
    return lines


def _command_line(ctx: SbatchContext) -> str:
    command: list[str] = []
    prefix = ctx.partition.launcher.get("command_prefix")
    if prefix:
        command.extend(shlex.split(str(prefix)))
    command.extend(["python", "-u", ctx.executable])
    command.extend(ctx.sweep_job.positional_args)
    overrides = [*ctx.sweep_job.hydra_overrides]
    overrides.extend(_injected_overrides(ctx))
    command.extend(overrides)
    return " ".join(shlex.quote(str(part)) for part in command)


def _injected_overrides(ctx: SbatchContext) -> list[str]:
    overrides: list[str] = []
    for key, value in ctx.partition.default_overrides.items():
        overrides.append(f"{key}={value}")
    if ctx.controls.batch is not None:
        if ctx.controls.batch % ctx.controls.gpu != 0:
            raise ValueError("BATCH must be divisible by GPU")
        overrides.append(f"data.batch_size={ctx.controls.batch // ctx.controls.gpu}")
    if ctx.resources.total_gpus > 1:
        for key, value in ctx.partition.resources.multi_gpu_overrides.items():
            overrides.append(f"{key}={value}")
    overrides.append(f"trainer.devices={ctx.resources.gpus_per_node}")
    overrides.append(f"trainer.num_nodes={ctx.resources.nodes}")
    overrides.append(f"hydra.run.dir={ctx.paths.run_dir}")
    wandb = ctx.partition.environment.wandb
    if wandb.get("set_name"):
        overrides.append(f"wandb.name={ctx.paths.display_name}")
    if wandb.get("set_group"):
        overrides.append(f"wandb.group={ctx.paths.run_root}")
    return overrides


def _slurm_value(value: Any) -> str:
    return str(value).replace("\n", " ")
