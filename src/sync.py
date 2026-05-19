"""Remote launcher synchronization."""

from __future__ import annotations

import hashlib
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from src.commands import CommandError, CommandRunner


class SyncError(RuntimeError):
    """Raised when local/remote launcher versions cannot be synchronized."""


STATE_FILE = ".newrunner-sync-state"


@dataclass(frozen=True)
class SyncResult:
    """Summary of a version synchronization check."""

    local_commit: str
    remote_commit: str | None
    updated: bool = False
    method: str = "state"


@dataclass(frozen=True)
class LocalGitState:
    """Local launcher git state."""

    commit: str
    dirty: bool
    fingerprint: str

    @property
    def state_text(self) -> str:
        """Serializable state stored on the remote snapshot."""

        dirty = "1" if self.dirty else "0"
        return f"commit={self.commit}\ndirty={dirty}\nfingerprint={self.fingerprint}\n"


def get_local_commit(runner: CommandRunner, *, cwd: str | None = None) -> str:
    """Return the local git commit hash."""

    return get_local_git_state(runner, cwd=cwd).commit


def get_local_git_state(runner: CommandRunner, *, cwd: str | None = None) -> LocalGitState:
    """Return local git state and a fingerprint of dirty contents."""

    commit = runner.run(["git", "rev-parse", "HEAD"], cwd=cwd).stdout.strip()
    status = runner.run(["git", "status", "--porcelain"], cwd=cwd).stdout
    dirty = bool(status.strip())
    fingerprint = _fingerprint_worktree(runner, commit=commit, status=status, cwd=cwd)
    return LocalGitState(commit=commit, dirty=dirty, fingerprint=fingerprint)


def sync_remote_launcher(
    *,
    runner: CommandRunner,
    remote_host: str,
    remote_launcher_dir: str,
    local_cwd: str | None = None,
) -> SyncResult:
    """Ensure the remote launcher snapshot matches the local working tree.

    Synchronization never relies on the remote host being able to reach a git
    server. Instead, the local working tree is rsynced when the remote state file
    does not match the local commit/dirty fingerprint.
    """

    local = get_local_git_state(runner, cwd=local_cwd)
    if local.dirty:
        print(
            "Warning: local checkout is dirty; remote launcher will be based on "
            "the current local working tree snapshot.",
            file=sys.stderr,
        )

    remote_state = _read_remote_state(runner, remote_host, remote_launcher_dir)
    if remote_state == local.state_text:
        return SyncResult(
            local_commit=local.commit,
            remote_commit=_commit_from_state(remote_state),
            updated=False,
        )

    _rsync_launcher(runner, remote_host, remote_launcher_dir, local_cwd)
    _write_remote_state(runner, remote_host, remote_launcher_dir, local.state_text)
    return SyncResult(
        local_commit=local.commit,
        remote_commit=_commit_from_state(remote_state),
        updated=True,
        method="rsync",
    )


def _fingerprint_worktree(
    runner: CommandRunner,
    *,
    commit: str,
    status: str,
    cwd: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(b"\0")
    digest.update(status.encode())
    digest.update(b"\0")
    diff = runner.run(["git", "diff", "HEAD", "--binary"], cwd=cwd).stdout
    digest.update(diff.encode())
    untracked = runner.run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=cwd).stdout
    root = Path(cwd or ".").resolve()
    for relative in sorted(part for part in untracked.split("\0") if part):
        path = root / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_remote_state(runner: CommandRunner, remote_host: str, remote_launcher_dir: str) -> str | None:
    state_path = shlex.quote(str(Path(remote_launcher_dir) / STATE_FILE))
    try:
        result = runner.run(["ssh", remote_host, f"cat {state_path}"])
    except CommandError:
        return None
    return result.stdout


def _write_remote_state(
    runner: CommandRunner,
    remote_host: str,
    remote_launcher_dir: str,
    state_text: str,
) -> None:
    state_path = shlex.quote(str(Path(remote_launcher_dir) / STATE_FILE))
    runner.run(["ssh", remote_host, f"cat > {state_path}"], input=state_text)


def _commit_from_state(state_text: str | None) -> str | None:
    if state_text is None:
        return None
    for line in state_text.splitlines():
        key, sep, value = line.partition("=")
        if sep and key == "commit":
            return value
    return None


def _rsync_launcher(
    runner: CommandRunner,
    remote_host: str,
    remote_launcher_dir: str,
    local_cwd: str | None,
) -> None:
    """Copy the local launcher working tree to the remote launcher directory."""

    source = Path(local_cwd or ".").resolve()
    runner.run(["ssh", remote_host, f"mkdir -p {shlex.quote(remote_launcher_dir)}"])
    runner.run(
        [
            "rsync",
            "-az",
            "--delete",
            "--exclude",
            ".git/",
            "--exclude",
            ".venv/",
            f"{source}/",
            f"{remote_host}:{remote_launcher_dir}/",
        ]
    )
