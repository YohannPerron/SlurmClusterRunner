"""Run directory planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Mapping

from newrunner.args import ControlParams
from newrunner.models import PartitionConfig
from newrunner.sweep import SweepJob


@dataclass(frozen=True)
class RunPathPlan:
    """Remote run paths for one expanded job."""

    run_root: str
    run_dir: str
    script_path: str
    stdout_path: str
    stderr_path: str
    job_id_path: str
    display_name: str


def invocation_stamp(now: datetime | None = None) -> str:
    """Return the timestamp component shared by all jobs in one invocation."""

    return (now or datetime.now()).strftime("%Y%m%d-%H%M")


def plan_run_paths(
    partition: PartitionConfig,
    executable: str,
    controls: ControlParams,
    sweep_job: SweepJob,
    *,
    index: int,
    timestamp: str,
    control_variable_params: Mapping[str, str] | None = None,
) -> RunPathPlan:
    """Build deterministic remote paths for one job.

    Layout:
    ``<log_dir>/<task>/<timestamp>[-NAME][-var_names]/<i>_[values]/``.
    """

    task_name = sanitize_path_component(Path(executable).stem or "job")
    name = sanitize_path_component(controls.name) if controls.name else None
    variable_params = dict(control_variable_params or {})
    variable_params.update(sweep_job.variable_params)

    run_root_name = timestamp
    if name:
        run_root_name += f"-{name}"
    named_variable_params = {
        key: value for key, value in variable_params.items() if not _is_positional_axis_name(key)
    }
    if named_variable_params:
        names = "_".join(sanitize_path_component(key) for key in named_variable_params)
        run_root_name += f"-{names}"

    value_suffix = "_".join(
        _display_param_component(key, value) for key, value in variable_params.items()
    )
    display_name = f"{index}"
    if value_suffix:
        display_name += f"_{value_suffix}"

    run_root = str(PurePosixPath(partition.paths.log_dir) / task_name / run_root_name)
    run_dir = str(PurePosixPath(run_root) / sanitize_path_component(display_name))
    return RunPathPlan(
        run_root=run_root,
        run_dir=run_dir,
        script_path=str(PurePosixPath(run_dir) / "run.sbatch"),
        stdout_path=str(PurePosixPath(run_dir) / "slurm-%j.out"),
        stderr_path=str(PurePosixPath(run_dir) / "slurm-%j.err"),
        job_id_path=str(PurePosixPath(run_dir) / "job_id.txt"),
        display_name=display_name,
    )


def _display_param_component(key: str, value: str) -> str:
    sanitized_value = sanitize_path_component(value)
    if _is_positional_axis_name(key):
        return sanitized_value
    return f"{sanitize_path_component(key)}={sanitized_value}"


def _is_positional_axis_name(key: str) -> bool:
    return key.startswith("arg") and key[3:].isdigit()


def sanitize_path_component(value: object) -> str:
    """Return a filesystem-safe, compact path component."""

    text = str(value).strip()
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9._=+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "value"
