from __future__ import annotations

import pytest

from newrunner.args import (
    ArgumentError,
    ControlParams,
    parse_cli,
    split_control_params,
    split_control_param_sweep,
    validate_control_params,
)
from newrunner.config import load_partitions


def test_parse_cli_shape():
    raw = parse_cli(["/proj", "train.py", "GPU=2", "lr=1e-3"])

    assert raw.project_path == "/proj"
    assert raw.executable == "train.py"
    assert raw.tokens == ["GPU=2", "lr=1e-3"]


def test_control_params_removed_from_forwarded_args():
    controls, forwarded = split_control_params(
        [
            "NAME=exp",
            "GPU=4",
            "PARTITION=jz-a100",
            "BATCH=128",
            "TIME=02:00:00",
            "MINTIME=true",
            "DEV=yes",
            "CONDA_ENV=env",
            "TAG=tag",
            "lr=1e-3",
            "checkpoint.ckpt",
        ]
    )

    assert controls.name == "exp"
    assert controls.gpu == 4
    assert controls.partition == "jz-a100"
    assert controls.batch == 128
    assert controls.time == "02:00:00"
    assert controls.mintime is True
    assert controls.dev is True
    assert controls.conda_env == "env"
    assert controls.tag == "tag"
    assert forwarded == ["lr=1e-3", "checkpoint.ckpt"]


def test_defaults_applied():
    controls, forwarded = split_control_params(["lr=1e-3"])

    assert controls == ControlParams(gpu=1, dev=False)
    assert forwarded == ["lr=1e-3"]


def test_control_params_can_sweep_gpu_and_batch():
    plan, forwarded = split_control_param_sweep(["GPU=2,4", "BATCH=64,128", "lr=1e-3"])

    assert forwarded == ["lr=1e-3"]
    assert [(controls.gpu, controls.batch) for controls in plan.controls] == [
        (2, 64),
        (2, 128),
        (4, 64),
        (4, 128),
    ]
    assert plan.variable_params == [
        {"GPU": "2", "BATCH": "64"},
        {"GPU": "2", "BATCH": "128"},
        {"GPU": "4", "BATCH": "64"},
        {"GPU": "4", "BATCH": "128"},
    ]


def test_compat_control_parser_rejects_sweep():
    with pytest.raises(ArgumentError, match="sweeps"):
        split_control_params(["GPU=2,4"])


@pytest.mark.parametrize("token, match", [("GPU=nope", "GPU"), ("BATCH=0", "BATCH"), ("DEV=maybe", "DEV")])
def test_invalid_types_fail_clearly(token, match):
    with pytest.raises(ArgumentError, match=match):
        split_control_params([token])


@pytest.mark.parametrize("time", ["2", "02:30:00"])
def test_time_accepts_integer_hours_or_hhmmss(time):
    controls, _ = split_control_params([f"TIME={time}"])

    assert controls.time == time


def test_invalid_time_fails_clearly():
    with pytest.raises(ArgumentError, match="TIME"):
        split_control_params(["TIME=2h"])


def test_partition_and_gpu_type_conflict_fails():
    with pytest.raises(ArgumentError, match="PARTITION"):
        split_control_params(["PARTITION=a", "GPU_TYPE=b"])


def test_tag_required_when_partition_requires_tag(tmp_path):
    text = """
name: p1
gpu_type: A100
remote_host: host
default: true
slurm:
  account: acc
  partition: part
  qos: qos
  dev_qos: dev
  max_time_hours: 20
  require_tag: true
resources:
  gpu_per_node: 8
  cpu_per_gpu: 10
  task_mode: per_gpu
paths:
  remote_launcher_dir: /remote/launcher
  default_project_dir: /remote/project
  data_dir: /remote/data
  log_dir: /remote/logs
environment:
  conda_env: env
  exports: {}
"""
    (tmp_path / "p1.yaml").write_text(text, encoding="utf-8")
    partition = load_partitions(tmp_path)[0]

    with pytest.raises(ArgumentError, match="TAG"):
        validate_control_params(ControlParams(), partition)

    validate_control_params(ControlParams(tag="baseline"), partition)
