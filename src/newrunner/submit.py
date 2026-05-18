"""Remote submission helpers."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from newrunner.commands import CommandRunner

_JOB_ID_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)")


class SubmissionError(RuntimeError):
    """Raised when remote submission cannot be completed."""


@dataclass(frozen=True)
class SubmissionResult:
    """Result of one submitted job."""

    run_dir: str
    job_id: str
    sbatch_path: str


def parse_sbatch_job_id(output: str) -> str:
    """Extract a SLURM job id from ``sbatch`` output."""

    match = _JOB_ID_RE.search(output)
    if not match:
        raise SubmissionError(f"Could not parse job id from sbatch output: {output.strip()!r}")
    return match.group(1)


def submit_sbatch(
    *,
    runner: CommandRunner,
    remote_host: str,
    run_dir: str,
    script: str,
    script_name: str = "run.sbatch",
) -> SubmissionResult:
    """Create a remote run directory, write an sbatch file, submit it, and save job id."""

    remote_run_dir = PurePosixPath(run_dir)
    sbatch_path = str(remote_run_dir / script_name)
    raw_output_path = str(remote_run_dir / "sbatch_output.txt")
    job_id_path = str(remote_run_dir / "job_id.txt")

    runner.run(["ssh", remote_host, f"mkdir -p {shlex.quote(str(remote_run_dir))}"])
    runner.run(["ssh", remote_host, f"cat > {shlex.quote(sbatch_path)}"], input=script)
    result = runner.run(
        ["ssh", remote_host, f"cd {shlex.quote(str(remote_run_dir))} && sbatch {shlex.quote(script_name)}"],
    )
    output = f"{result.stdout}{result.stderr}"
    try:
        job_id = parse_sbatch_job_id(output)
    except SubmissionError:
        runner.run(["ssh", remote_host, f"cat > {shlex.quote(raw_output_path)}"], input=output, check=False)
        raise
    runner.run(["ssh", remote_host, f"cat > {shlex.quote(job_id_path)}"], input=f"{job_id}\n")
    return SubmissionResult(run_dir=str(remote_run_dir), job_id=job_id, sbatch_path=sbatch_path)
