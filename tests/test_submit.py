from __future__ import annotations

import pytest

from newrunner.commands import CommandResult
from newrunner.submit import SubmissionError, parse_sbatch_job_id, submit_sbatch


class FakeRunner:
    def __init__(self, sbatch_output: str = "Submitted batch job 123456\n") -> None:
        self.calls: list[tuple[tuple[str, ...], str | None, bool]] = []
        self.sbatch_output = sbatch_output

    def run(self, args, *, cwd=None, input=None, check=True):
        del cwd
        command = tuple(args)
        self.calls.append((command, input, check))
        if "sbatch" in command[-1]:
            return CommandResult(command, 0, stdout=self.sbatch_output)
        return CommandResult(command, 0)


def test_parse_sbatch_job_id() -> None:
    assert parse_sbatch_job_id("Submitted batch job 42") == "42"


def test_parse_sbatch_job_id_fails() -> None:
    with pytest.raises(SubmissionError):
        parse_sbatch_job_id("no job here")


def test_submit_sbatch_writes_script_submits_and_records_job_id() -> None:
    runner = FakeRunner()

    result = submit_sbatch(
        runner=runner,
        remote_host="cluster",
        run_dir="/logs/run0",
        script="#!/bin/bash\n",
    )

    assert result.job_id == "123456"
    assert result.sbatch_path == "/logs/run0/run.sbatch"
    assert runner.calls[0][0] == ("ssh", "cluster", "mkdir -p /logs/run0")
    assert runner.calls[1] == (("ssh", "cluster", "cat > /logs/run0/run.sbatch"), "#!/bin/bash\n", True)
    assert runner.calls[2][0] == ("ssh", "cluster", "cd /logs/run0 && sbatch run.sbatch")
    assert runner.calls[3] == (("ssh", "cluster", "cat > /logs/run0/job_id.txt"), "123456\n", True)


def test_submit_sbatch_saves_raw_output_when_job_id_missing() -> None:
    runner = FakeRunner("unexpected output")

    with pytest.raises(SubmissionError):
        submit_sbatch(runner=runner, remote_host="cluster", run_dir="/logs/run0", script="script")

    assert runner.calls[-1] == (
        ("ssh", "cluster", "cat > /logs/run0/sbatch_output.txt"),
        "unexpected output",
        False,
    )
