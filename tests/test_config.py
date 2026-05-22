from __future__ import annotations

import pytest

from src.config import ConfigError, default_partitions_dir, load_partitions, select_partition


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
  gpu_per_node: 8
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


def write_partition(tmp_path, name="p1", text=BASE):
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_partition_loads(tmp_path):
    write_partition(tmp_path)

    partitions = load_partitions(tmp_path)

    assert len(partitions) == 1
    partition = partitions[0]
    assert partition.name == "p1"
    assert partition.remote_host == "host"
    assert partition.resources.gpu_per_node == 8
    assert partition.paths.log_dir == "/remote/logs"
    assert partition.environment.activate_command is None


def test_activate_command_loads_and_can_be_null(tmp_path):
    write_partition(tmp_path, text=BASE.replace("  exports: {}", "  activate_command: null\n  exports: {}"))

    partition = load_partitions(tmp_path)[0]

    assert partition.environment.activate_command is None


def test_activate_command_loads(tmp_path):
    write_partition(tmp_path, text=BASE.replace("  exports: {}", "  activate_command: env activate custom\n  exports: {}"))

    partition = load_partitions(tmp_path)[0]

    assert partition.environment.activate_command == "env activate custom"


def test_missing_required_field_fails(tmp_path):
    write_partition(tmp_path, text=BASE.replace("remote_host: host\n", ""))

    with pytest.raises(ConfigError, match="remote_host"):
        load_partitions(tmp_path)


def test_ambiguous_defaults_fail(tmp_path):
    write_partition(tmp_path, "p1", BASE)
    write_partition(tmp_path, "p2", BASE.replace("name: p1", "name: p2"))

    partitions = load_partitions(tmp_path)

    with pytest.raises(ConfigError, match="Multiple default"):
        select_partition(partitions)


def test_no_default_fails(tmp_path):
    write_partition(tmp_path, text=BASE.replace("default: true", "default: false"))

    partitions = load_partitions(tmp_path)

    with pytest.raises(ConfigError, match="no default"):
        select_partition(partitions)


def test_partition_selection_works(tmp_path):
    write_partition(tmp_path, "p1", BASE.replace("default: true", "default: false"))
    write_partition(tmp_path, "p2", BASE.replace("name: p1", "name: p2").replace("default: true", "default: false"))

    partition = select_partition(load_partitions(tmp_path), partition_name="p2")

    assert partition.name == "p2"


def test_gpu_type_emits_warning(tmp_path):
    write_partition(tmp_path)

    with pytest.warns(DeprecationWarning, match="GPU_TYPE"):
        partition = select_partition(load_partitions(tmp_path), gpu_type="p1")

    assert partition.name == "p1"


def test_project_partitions_load():
    partitions = load_partitions(default_partitions_dir())

    assert {partition.name for partition in partitions} >= {"jz-a100", "ada-mi300"}


def test_default_partition_dir_is_not_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    partitions = load_partitions()

    assert {partition.name for partition in partitions} >= {"jz-a100", "ada-mi300"}
