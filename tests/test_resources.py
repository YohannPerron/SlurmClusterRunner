from __future__ import annotations

import pytest

from src.config import load_partitions
from src.resources import ResourceError, calculate_resources


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
  exports: {}
"""


def partition(tmp_path, resources: str = ""):
    text = BASE if not resources else BASE.replace(
        "resources:\n  gpu_per_node: 4\n  cpu_per_gpu: 10\n  task_mode: per_gpu",
        resources,
    )
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    return load_partitions(tmp_path)[0]


def test_single_gpu(tmp_path):
    req = calculate_resources(1, partition(tmp_path))
    assert req.nodes == 1
    assert req.gpus_per_node == 1
    assert req.ntasks_per_node == 1
    assert req.cpus_per_task == 10


def test_multi_gpu_one_node(tmp_path):
    req = calculate_resources(3, partition(tmp_path))
    assert req.nodes == 1
    assert req.gpus_per_node == 3
    assert req.ntasks_per_node == 3


def test_multi_node_valid(tmp_path):
    req = calculate_resources(8, partition(tmp_path))
    assert req.nodes == 2
    assert req.gpus_per_node == 4
    assert req.ntasks_per_node == 4


def test_multi_node_invalid_divisibility(tmp_path):
    with pytest.raises(ResourceError, match="divisible"):
        calculate_resources(6, partition(tmp_path))


def test_per_node_task_mode(tmp_path):
    part = partition(tmp_path, "resources:\n  gpu_per_node: 4\n  cpu_per_gpu: 10\n  task_mode: per_node")
    req = calculate_resources(3, part)
    assert req.ntasks_per_node == 1
    assert req.cpus_per_task == 30


def test_require_full_node_gpu(tmp_path):
    part = partition(
        tmp_path,
        "resources:\n  gpu_per_node: 4\n  require_full_node_gpu: true\n  cpu_per_gpu: 10\n  task_mode: per_gpu",
    )
    with pytest.raises(ResourceError, match="full-node"):
        calculate_resources(1, part)
    assert calculate_resources(4, part).gpus_per_node == 4
