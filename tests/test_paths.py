from datetime import datetime

from newrunner.args import ControlParams
from newrunner.models import EnvironmentConfig, PartitionConfig, PathsConfig, ResourcesConfig, SlurmConfig
from newrunner.paths import invocation_stamp, plan_run_paths, sanitize_path_component
from newrunner.sweep import parse_sweep


def partition() -> PartitionConfig:
    return PartitionConfig(
        name="test",
        remote_host="host",
        gpu_type="a100",
        slurm=SlurmConfig(account=None, partition=None, qos=None, dev_qos=None, max_time_hours=1),
        resources=ResourcesConfig(gpu_per_node=4, cpu_per_gpu=8, task_mode="per_gpu"),
        paths=PathsConfig(remote_launcher_dir="/launcher", default_project_dir="/proj", data_dir="/data", log_dir="/logs"),
        environment=EnvironmentConfig(conda_env="env"),
    )


def test_invocation_stamp_is_minute_resolution() -> None:
    assert invocation_stamp(datetime(2026, 5, 18, 9, 7, 33)) == "20260518-0907"


def test_variable_params_included_in_root_and_job_dir() -> None:
    job = parse_sweep(["lr=1e-3,1e-4", "model=resnet"]).jobs[0]
    plan = plan_run_paths(
        partition(),
        "train.py",
        ControlParams(name="my run"),
        job,
        index=0,
        timestamp="20260518-0907",
    )

    assert plan.run_root == "/logs/train/20260518-0907-my_run-lr"
    assert plan.run_dir == "/logs/train/20260518-0907-my_run-lr/0_lr=1e-3"
    assert plan.script_path.endswith("/run.sbatch")
    assert plan.job_id_path.endswith("/job_id.txt")


def test_positional_sweep_values_are_sanitized_but_not_named() -> None:
    job = parse_sweep(["data/a.csv,data/b.csv"]).jobs[1]
    plan = plan_run_paths(partition(), "scripts/run train.py", ControlParams(), job, index=1, timestamp="ts")

    assert plan.run_root == "/logs/run_train/ts"
    assert plan.display_name == "1_data_b.csv"


def test_control_variable_params_included() -> None:
    job = parse_sweep(["lr=3e-4"]).jobs[0]
    plan = plan_run_paths(
        partition(), "train.py", ControlParams(gpu=4), job, index=2, timestamp="ts", control_variable_params={"GPU": "4"}
    )

    assert plan.run_root == "/logs/train/ts-GPU"
    assert plan.display_name == "2_GPU=4"


def test_sanitize_empty_component() -> None:
    assert sanitize_path_component(" /// ") == "value"
