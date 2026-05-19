"""Manual integration test for submitting a tiny job on Jean Zay A100.

This test submits a real SLURM job through SSH and is intentionally disabled by
default. Run it only when you are on a machine configured to reach the ``JZ`` SSH
host and you really want to submit a job:

    NEWRUNNER_RUN_JZ_A100_SUBMIT=1 uv run pytest tests/test_manual_jz_a100_submit.py -s
"""

from __future__ import annotations

import os
import shlex
from datetime import datetime
from pathlib import PurePosixPath

import pytest

from src.cli import orchestrate
from src.commands import SubprocessRunner
from src.config import load_selected_partition
from src.submit import submit_sbatch

pytestmark = pytest.mark.skipif(
    os.environ.get("NEWRUNNER_RUN_JZ_A100_SUBMIT") != "1",
    reason="manual test: set NEWRUNNER_RUN_JZ_A100_SUBMIT=1 to submit a real JZ A100 job",
)


def test_manual_submit_dummy_work_on_jz_a100() -> None:
    """Submit a tiny real job to the configured jz-a100 partition."""

    print("[manual-jz-a100] loading partition jz-a100", flush=True)
    partition = load_selected_partition(partition_name="jz-a100")
    print(f"[manual-jz-a100] remote host: {partition.remote_host}", flush=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = str(PurePosixPath(partition.paths.log_dir) / "manual-newrunner-tests" / stamp)
    qos = partition.slurm.dev_qos or partition.slurm.qos

    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=newrunner-dummy",
        f"#SBATCH --account={partition.slurm.account}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --constraint={partition.slurm.constraint}",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={partition.resources.cpu_per_gpu}",
        "#SBATCH --time=00:05:00",
        f"#SBATCH --output={run_dir}/slurm-%j.out",
        f"#SBATCH --error={run_dir}/slurm-%j.out",
        "",
        "set -euo pipefail",
        "hostname",
        "date",
        "nvidia-smi || true",
        "echo newrunner dummy job completed",
    ]
    if partition.slurm.partition:
        script_lines.insert(4, f"#SBATCH --partition={partition.slurm.partition}")

    runner = SubprocessRunner()
    script = "\n".join(script_lines) + "\n"
    print(f"[manual-jz-a100] submitting tiny sbatch in {run_dir}", flush=True)
    result = submit_sbatch(
        runner=runner,
        remote_host=partition.remote_host,
        run_dir=run_dir,
        script=script,
    )

    print(f"[manual-jz-a100] submitted job {result.job_id}; verifying remote sbatch", flush=True)
    sbatch_file = PurePosixPath(result.sbatch_path)
    runner.run(["ssh", partition.remote_host, f"test -s {shlex.quote(str(sbatch_file))}"])
    print(f"[manual-jz-a100] reading remote sbatch {sbatch_file}", flush=True)
    remote_script = runner.run(["ssh", partition.remote_host, f"cat {shlex.quote(str(sbatch_file))}"]).stdout

    assert remote_script == script
    assert result.job_id.isdigit()
    print(f"Submitted JZ A100 dummy job {result.job_id} in {result.run_dir}")


def test_manual_full_pipeline_dummy_sweep_on_jz_a100() -> None:
    """Run the full NewRunner pipeline for a larger dummy sweep on JZ A100.

    This exercises CLI parsing, partition loading, remote launcher sync, control
    parameters, positional and Hydra sweeps, bracket-aware values, resource/time
    planning, sbatch rendering, submission, and remote job-id writing. It
    requires a clean local git checkout because the sync step intentionally
    refuses dirty submissions.
    """

    print("[manual-jz-a100-full] loading partition jz-a100", flush=True)
    partition = load_selected_partition(partition_name="jz-a100")
    print(f"[manual-jz-a100-full] remote host: {partition.remote_host}", flush=True)
    runner = SubprocessRunner()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    remote_project = str(
        PurePosixPath(partition.paths.log_dir)
        / "manual-newrunner-tests"
        / f"full-pipeline-project-{stamp}"
    )
    remote_script = PurePosixPath(remote_project) / "dummy_train.py"
    script_source = """#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

print(\"newrunner full-pipeline dummy\")
print(\"argv=\" + repr(sys.argv[1:]))
run_dir = None
for arg in sys.argv[1:]:
    if arg.startswith(\"hydra.run.dir=\"):
        run_dir = arg.split(\"=\", 1)[1]
if run_dir:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / \"dummy_done.txt\").write_text(\"ok\\n\")
"""

    print(f"[manual-jz-a100-full] creating remote project {remote_project}", flush=True)
    runner.run(["ssh", partition.remote_host, f"mkdir -p {shlex.quote(remote_project)}"])
    print(f"[manual-jz-a100-full] writing remote dummy script {remote_script}", flush=True)
    runner.run(
        ["ssh", partition.remote_host, f"cat > {shlex.quote(str(remote_script))}"],
        input=script_source,
    )

    print("[manual-jz-a100-full] starting orchestrate: sync + render + submit 8 jobs", flush=True)
    summaries = orchestrate(
        [
            remote_project,
            "dummy_train.py",
            "PARTITION=jz-a100",
            "NAME=manual-full-pipeline",
            "GPU=1",
            "BATCH=8,16",
            "MINTIME=false",
            "DEV=true",
            "TIME=00:05:00",
            "TAG=newrunner-manual-full-pipeline",
            "pos_a,pos_b",
            "dummy.value=1,2",
            "dummy.list=[1,2,3]",
            "dummy.flag=true",
        ],
        runner=runner,
    )

    print(f"[manual-jz-a100-full] orchestrate returned {len(summaries)} summaries", flush=True)
    assert len(summaries) == 8
    assert {summary.submission.job_id.isdigit() for summary in summaries} == {True}
    for i, summary in enumerate(summaries, start=1):
        print(
            f"[manual-jz-a100-full] verifying job {i}/8: "
            f"{summary.submission.job_id}",
            flush=True,
        )
        runner.run(
            ["ssh", partition.remote_host, f"test -s {shlex.quote(summary.submission.sbatch_path)}"]
        )
        runner.run(
            ["ssh", partition.remote_host, f"test -s {shlex.quote(summary.paths.job_id_path)}"]
        )
        print(
            "Submitted full-pipeline JZ A100 dummy job "
            f"{summary.submission.job_id} in {summary.submission.run_dir}"
        )
