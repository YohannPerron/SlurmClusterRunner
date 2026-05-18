from __future__ import annotations

import pytest

from newrunner.commands import CommandResult
from newrunner.sync import SyncError, sync_remote_launcher


class FakeRunner:
    def __init__(self, *, dirty: bool = False, remote_commits: list[str] | None = None) -> None:
        self.dirty = dirty
        self.remote_commits = remote_commits or ["abc"]
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def run(self, args, *, cwd=None, input=None, check=True):
        del cwd, input, check
        command = tuple(args)
        self.calls.append((command, None))
        if command == ("git", "rev-parse", "HEAD"):
            return CommandResult(command, 0, stdout="abc\n")
        if command == ("git", "status", "--porcelain"):
            return CommandResult(command, 0, stdout=" M file.py\n" if self.dirty else "")
        remote_command = command[-1]
        if "git rev-parse HEAD" in remote_command:
            return CommandResult(command, 0, stdout=f"{self.remote_commits.pop(0)}\n")
        return CommandResult(command, 0)


def test_sync_remote_launcher_matching_version_does_nothing() -> None:
    runner = FakeRunner(remote_commits=["abc"])

    result = sync_remote_launcher(
        runner=runner,
        remote_host="cluster",
        remote_launcher_dir="/launcher",
    )

    assert result.updated is False
    assert len([call for call in runner.calls if "git checkout" in call[0][-1]]) == 0


def test_sync_remote_launcher_mismatch_fetches_and_checkouts_local_commit() -> None:
    runner = FakeRunner(remote_commits=["old", "abc"])

    result = sync_remote_launcher(
        runner=runner,
        remote_host="cluster",
        remote_launcher_dir="/launcher",
    )

    assert result.updated is True
    assert ("ssh", "cluster", "cd /launcher && git fetch && git checkout abc") in [call[0] for call in runner.calls]


def test_sync_remote_launcher_refuses_dirty_checkout() -> None:
    runner = FakeRunner(dirty=True)

    with pytest.raises(SyncError):
        sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")


def test_sync_remote_launcher_fails_when_remote_still_mismatches() -> None:
    runner = FakeRunner(remote_commits=["old", "still-old"])

    with pytest.raises(SyncError):
        sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")
