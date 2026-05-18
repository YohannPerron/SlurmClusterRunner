"""Remote launcher synchronization."""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from newrunner.commands import CommandRunner


class SyncError(RuntimeError):
    """Raised when local/remote launcher versions cannot be synchronized."""


@dataclass(frozen=True)
class SyncResult:
    """Summary of a version synchronization check."""

    local_commit: str
    remote_commit: str | None
    updated: bool = False


def get_local_commit(runner: CommandRunner, *, cwd: str | None = None) -> str:
    """Return the local git commit hash, refusing dirty working trees."""

    commit = runner.run(["git", "rev-parse", "HEAD"], cwd=cwd).stdout.strip()
    dirty = runner.run(["git", "status", "--porcelain"], cwd=cwd).stdout.strip()
    if dirty:
        raise SyncError("Local checkout is dirty; refusing remote submission")
    return commit


def sync_remote_launcher(
    *,
    runner: CommandRunner,
    remote_host: str,
    remote_launcher_dir: str,
    local_cwd: str | None = None,
) -> SyncResult:
    """Ensure the remote launcher git checkout matches the local commit."""

    local_commit = get_local_commit(runner, cwd=local_cwd)
    quoted_dir = shlex.quote(remote_launcher_dir)
    remote = runner.run(
        ["ssh", remote_host, f"cd {quoted_dir} && git rev-parse HEAD"],
    ).stdout.strip()
    if remote == local_commit:
        return SyncResult(local_commit=local_commit, remote_commit=remote, updated=False)

    runner.run(["ssh", remote_host, f"cd {quoted_dir} && git fetch && git checkout {shlex.quote(local_commit)}"])
    synced = runner.run(
        ["ssh", remote_host, f"cd {quoted_dir} && git rev-parse HEAD"],
    ).stdout.strip()
    if synced != local_commit:
        raise SyncError(
            "Remote launcher synchronization failed: "
            f"local={local_commit} remote_after_sync={synced}"
        )
    return SyncResult(local_commit=local_commit, remote_commit=remote, updated=True)
