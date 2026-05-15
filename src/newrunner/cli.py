"""Command-line interface for NewRunner."""

from __future__ import annotations

import sys

from newrunner.args import ArgumentError, parse_cli, split_control_params, validate_control_params
from newrunner.config import ConfigError, load_selected_partition


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``newrunner`` command."""

    raw = parse_cli(sys.argv[1:] if argv is None else argv)
    controls, forwarded = split_control_params(raw.tokens)
    try:
        partition = load_selected_partition(
            partition_name=controls.partition,
            gpu_type=controls.gpu_type,
        )
        validate_control_params(controls, partition)
    except (ArgumentError, ConfigError) as exc:
        raise SystemExit(str(exc)) from exc

    # Full orchestration is implemented in later plan steps. For now, this
    # proves parsing and selection while avoiding accidental submission.
    print(f"Partition: {partition.name}")
    print(f"Project: {raw.project_path}")
    print(f"Executable: {raw.executable}")
    print(f"Forwarded args: {' '.join(forwarded)}")
