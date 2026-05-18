"""Command-line interface for NewRunner."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from newrunner.args import ArgumentError, CONTROL_PARAMS, parse_cli, split_control_params, validate_control_params
from newrunner.commands import CommandError, CommandRunner, SubprocessRunner
from newrunner.config import ConfigError, load_selected_partition
from newrunner.paths import RunPathPlan, invocation_stamp, plan_run_paths
from newrunner.resources import ResourceError, calculate_resources
from newrunner.sbatch import SbatchContext, render_sbatch
from newrunner.submit import SubmissionError, SubmissionResult, submit_sbatch
from newrunner.sweep import SweepConfirmationRequired, SweepJob, parse_sweep
from newrunner.sync import SyncError, sync_remote_launcher
from newrunner.time import TimeError, resolve_time


@dataclass(frozen=True)
class JobSummary:
    """Submitted job data printed at the end of an invocation."""

    partition_name: str
    remote_host: str
    gpu: int
    paths: RunPathPlan
    submission: SubmissionResult


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``newrunner`` command."""

    try:
        summaries = orchestrate(sys.argv[1:] if argv is None else argv, runner=SubprocessRunner())
    except (
        ArgumentError,
        ConfigError,
        ResourceError,
        TimeError,
        SubmissionError,
        SyncError,
        CommandError,
        SweepConfirmationRequired,
        ValueError,
    ) as exc:
        if isinstance(exc, SweepConfirmationRequired):
            names = ", ".join(exc.control_names)
            message = (
                "Sweeping control parameter(s) "
                f"{names} requires --allow-control-sweeps. "
                "GPU, PARTITION, and BATCH may be swept without this flag."
            )
        else:
            message = str(exc)
        raise SystemExit(message) from exc
    print_summary(summaries)


def orchestrate(argv: list[str], *, runner: CommandRunner) -> list[JobSummary]:
    """Run the full parse/sync/render/submit pipeline."""

    raw = parse_cli(argv)
    sweep = parse_sweep(raw.tokens, confirm_control_sweeps=raw.allow_control_sweeps)
    timestamp = invocation_stamp()
    jobs: list[tuple[SbatchContext, RunPathPlan]] = []
    synced: set[tuple[str, str]] = set()

    for index, sweep_job in enumerate(sweep.jobs):
        controls, forwarded = split_control_params(sweep_job.tokens)
        partition = load_selected_partition(partition_name=controls.partition, gpu_type=controls.gpu_type)
        validate_control_params(controls, partition)

        sync_key = (partition.remote_host, partition.paths.remote_launcher_dir)
        if sync_key not in synced:
            sync_remote_launcher(
                runner=runner,
                remote_host=partition.remote_host,
                remote_launcher_dir=partition.paths.remote_launcher_dir,
            )
            synced.add(sync_key)

        clean_job = _clean_sweep_job(sweep_job, forwarded)
        resources = calculate_resources(controls.gpu, partition)
        time_request = resolve_time(controls, partition)
        control_vars = {
            key: value for key, value in sweep_job.variable_params.items() if key in CONTROL_PARAMS
        }
        paths = plan_run_paths(
            partition,
            raw.executable,
            controls,
            clean_job,
            index=index,
            timestamp=timestamp,
            control_variable_params=control_vars,
        )
        ctx = SbatchContext(
            partition=partition,
            project_path=raw.project_path or partition.paths.default_project_dir,
            executable=raw.executable,
            controls=controls,
            sweep_job=clean_job,
            resources=resources,
            time=time_request,
            paths=paths,
        )
        jobs.append((ctx, paths))

    summaries: list[JobSummary] = []
    for ctx, paths in jobs:
        script = render_sbatch(ctx)
        submission = submit_sbatch(
            runner=runner,
            remote_host=ctx.partition.remote_host,
            run_dir=paths.run_dir,
            script=script,
        )
        summaries.append(JobSummary(ctx.partition.name, ctx.partition.remote_host, ctx.controls.gpu, paths, submission))
    return summaries


def _clean_sweep_job(original: SweepJob, forwarded: list[str]) -> SweepJob:
    parsed = parse_sweep(forwarded).jobs[0] if forwarded else SweepJob([], [], [], {})
    variable_params = {
        key: value for key, value in original.variable_params.items() if key not in CONTROL_PARAMS
    }
    return SweepJob(
        tokens=parsed.tokens,
        positional_args=parsed.positional_args,
        hydra_overrides=parsed.hydra_overrides,
        variable_params=variable_params,
    )


def print_summary(summaries: list[JobSummary]) -> None:
    """Print a compact submission summary."""

    if not summaries:
        print("Jobs: 0")
        return
    first = summaries[0]
    print(f"Partition: {first.partition_name}")
    print(f"Jobs: {len(summaries)}")
    print(f"GPU/job: {first.gpu}")
    print(f"Remote host: {first.remote_host}")
    print(f"Run root: {first.paths.run_root}")
    print("Submitted:")
    for summary in summaries:
        print(f"  {summary.paths.display_name} -> {summary.submission.job_id}")
