from __future__ import annotations

from pathlib import Path

from newrunner.commands import CommandError, CommandResult
from newrunner.sync import sync_remote_launcher


class FakeRunner:
    def __init__(
        self,
        *,
        dirty: bool = False,
        remote_state: str | None = None,
        cat_fails: bool = False,
    ) -> None:
        self.dirty = dirty
        self.remote_state = remote_state
        self.cat_fails = cat_fails
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def run(self, args, *, cwd=None, input=None, check=True):
        del check
        command = tuple(args)
        self.calls.append((command, input))
        if command == ("git", "rev-parse", "HEAD"):
            return CommandResult(command, 0, stdout="abc\n")
        if command == ("git", "status", "--porcelain"):
            return CommandResult(command, 0, stdout=" M file.py\n" if self.dirty else "")
        if command == ("git", "diff", "HEAD", "--binary"):
            return CommandResult(command, 0, stdout="diff contents" if self.dirty else "")
        if command == ("git", "ls-files", "--others", "--exclude-standard", "-z"):
            return CommandResult(command, 0, stdout="")
        remote_command = command[-1]
        if remote_command.startswith("cat > "):
            return CommandResult(command, 0)
        if remote_command.startswith("cat "):
            if self.cat_fails:
                raise CommandError(CommandResult(command, 1, stderr="missing"))
            return CommandResult(command, 0, stdout=self.remote_state or "")
        return CommandResult(command, 0)


def local_state_text(dirty: bool = False) -> str:
    runner = FakeRunner(dirty=dirty)
    result = sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")
    state_writes = [call for call in runner.calls if call[0][-1] == "cat > /launcher/.newrunner-sync-state"]
    assert result.updated is True
    assert state_writes
    return state_writes[-1][1] or ""


def test_sync_remote_launcher_matching_state_does_nothing() -> None:
    state = local_state_text()
    runner = FakeRunner(remote_state=state)

    result = sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")

    assert result.updated is False
    assert not any(call[0][0] == "rsync" for call in runner.calls)


def test_sync_remote_launcher_mismatch_rsyncs_and_writes_state() -> None:
    runner = FakeRunner(remote_state="commit=old\ndirty=0\nfingerprint=old\n")

    result = sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")

    assert result.updated is True
    assert result.method == "rsync"
    assert any(call[0][0] == "rsync" for call in runner.calls)
    assert any(call[0][-1] == "cat > /launcher/.newrunner-sync-state" for call in runner.calls)


def test_sync_remote_launcher_missing_state_rsyncs() -> None:
    runner = FakeRunner(cat_fails=True)

    result = sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")

    assert result.updated is True
    assert any(call[0][0] == "rsync" for call in runner.calls)


def test_sync_remote_launcher_dirty_checkout_warns_and_rsyncs(capsys) -> None:
    runner = FakeRunner(dirty=True, remote_state=local_state_text(dirty=False))

    result = sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher")

    assert result.updated is True
    assert result.method == "rsync"
    assert "local checkout is dirty" in capsys.readouterr().err
    assert any(call[0][0] == "rsync" for call in runner.calls)


def test_dirty_fingerprint_includes_untracked_file(tmp_path: Path) -> None:
    (tmp_path / "new.txt").write_text("hello")

    class UntrackedRunner(FakeRunner):
        def run(self, args, *, cwd=None, input=None, check=True):
            if tuple(args) == ("git", "ls-files", "--others", "--exclude-standard", "-z"):
                return CommandResult(tuple(args), 0, stdout="new.txt\0")
            return super().run(args, cwd=cwd, input=input, check=check)

    runner = UntrackedRunner(dirty=True, cat_fails=True)
    sync_remote_launcher(runner=runner, remote_host="cluster", remote_launcher_dir="/launcher", local_cwd=str(tmp_path))

    state_writes = [call for call in runner.calls if call[0][-1] == "cat > /launcher/.newrunner-sync-state"]
    assert state_writes
