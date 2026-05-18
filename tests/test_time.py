from __future__ import annotations

import pytest

from newrunner.args import ControlParams
from newrunner.config import load_partitions
from newrunner.time import TimeError, parse_wall_time, resolve_time


BASE = """
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
  default_time: '19:59:59'
  dev_time: '01:00:00'
  min_time: '19:00:00'
  use_min_time_default: true
  require_tag: false
resources:
  gpu_per_node: 4
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
features:
  job_chaining: false
"""


def partition(tmp_path, text: str = BASE):
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    return load_partitions(tmp_path)[0]


def test_parse_wall_time_integer_hours_and_hms():
    assert parse_wall_time("2") == 7200
    assert parse_wall_time("01:02:03") == 3723


def test_default_time(tmp_path):
    req = resolve_time(ControlParams(), partition(tmp_path))
    assert req.wall_time == "19:59:59"
    assert req.use_min_time is True


def test_dev_time(tmp_path):
    req = resolve_time(ControlParams(dev=True), partition(tmp_path))
    assert req.wall_time == "01:00:00"


def test_requested_time_and_mintime_override(tmp_path):
    req = resolve_time(ControlParams(time="2", mintime=False), partition(tmp_path))
    assert req.wall_time == "02:00:00"
    assert req.use_min_time is False


def test_time_max_validation(tmp_path):
    with pytest.raises(TimeError, match="exceeds"):
        resolve_time(ControlParams(time="21:00:00"), partition(tmp_path))


def test_job_chaining_not_implemented_message(tmp_path):
    text = BASE.replace("job_chaining: false", "job_chaining: true")
    with pytest.raises(TimeError, match="not yet implemented"):
        resolve_time(ControlParams(time="21:00:00"), partition(tmp_path, text))
