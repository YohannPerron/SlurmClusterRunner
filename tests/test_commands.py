"""Tests for external command preparation."""

from __future__ import annotations

import subprocess

from src.commands import SubprocessRunner


def test_existing_ssh_master_is_reused_for_ssh_and_rsync(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="Master running\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SubprocessRunner(_control_dir=tmp_path)

    ssh_command = runner._prepare_command(["ssh", "Terra", "true"])
    rsync_command = runner._prepare_command(["rsync", "/local/", "Terra:/remote/"])
    runner.close_ssh_masters()

    assert ssh_command == ["ssh", "Terra", "true"]
    assert rsync_command == ["rsync", "/local/", "Terra:/remote/"]
    assert calls == [["ssh", "-O", "check", "Terra"]]


def test_missing_external_master_uses_and_closes_private_master(
    monkeypatch, tmp_path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        returncode = 255 if command[1:3] == ["-O", "check"] else 0
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = SubprocessRunner(_control_dir=tmp_path)

    prepared = runner._prepare_command(["ssh", "Terra", "true"])
    runner.close_ssh_masters()

    control_path = str(tmp_path / "cm-Terra")
    options = [
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=10m",
        "-o",
        f"ControlPath={control_path}",
    ]
    assert prepared == ["ssh", *options, "Terra", "true"]
    assert calls == [
        ["ssh", "-O", "check", "Terra"],
        ["ssh", *options, "-MNf", "-o", "ControlMaster=yes", "Terra"],
        ["ssh", *options, "-O", "exit", "Terra"],
    ]
