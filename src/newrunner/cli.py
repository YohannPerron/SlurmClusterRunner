"""Command-line interface for NewRunner."""

from __future__ import annotations

import sys

from newrunner.args import ArgumentError, parse_cli, split_control_params, validate_control_params
from newrunner.config import ConfigError, load_selected_partition
from newrunner.sweep import parse_sweep


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``newrunner`` command."""

    raw = parse_cli(sys.argv[1:] if argv is None else argv)
    sweep = parse_sweep(raw.tokens)
    parsed_jobs = []
    try:
        for job in sweep.jobs:
            controls, forwarded = split_control_params(job.tokens)
            partition = load_selected_partition(
                partition_name=controls.partition,
                gpu_type=controls.gpu_type,
            )
            validate_control_params(controls, partition)
            parsed_jobs.append((partition, controls, forwarded))
    except (ArgumentError, ConfigError) as exc:
        raise SystemExit(str(exc)) from exc

    # Full orchestration is implemented in later plan steps. For now, this
    # proves parsing and selection while avoiding accidental submission.
    first_partition = parsed_jobs[0][0]
    print(f"Partition: {first_partition.name}")
    print(f"Project: {raw.project_path}")
    print(f"Executable: {raw.executable}")
    print(f"Jobs: {len(parsed_jobs)}")
    for index, (_, controls, forwarded) in enumerate(parsed_jobs):
        print(f"Job {index}: GPU={controls.gpu} Forwarded args: {' '.join(forwarded)}")
