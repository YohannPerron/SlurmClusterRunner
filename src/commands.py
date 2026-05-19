"""Subprocess command abstraction."""

from __future__ import annotations

import atexit
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class CommandResult:
    """Completed command data."""

    args: Sequence[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandError(RuntimeError):
    """Raised when an external command fails."""

    def __init__(self, result: CommandResult) -> None:
        self.result = result
        command = " ".join(result.args)
        message = f"Command failed ({result.returncode}): {command}"
        details = (result.stderr or result.stdout).strip()
        if details:
            message = f"{message}\n{details}"
        super().__init__(message)


class CommandRunner(Protocol):
    """Interface used by submission/sync code and tests."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: str | None = None,
        input: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        """Run a command."""


@dataclass
class SubprocessRunner:
    """Command runner backed by :mod:`subprocess`.

    SSH calls are multiplexed through a persistent master connection per host.
    This avoids paying the SSH login cost for every mkdir/cat/sbatch command.
    """

    use_ssh_master: bool = True
    verbose: bool = False
    _control_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="runner-ssh-")))
    _masters: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.use_ssh_master:
            atexit.register(self.close_ssh_masters)

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: str | None = None,
        input: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        args = self._prepare_command(args)
        command = " ".join(shlex.quote(str(part)) for part in args)
        cwd_text = f" cwd={cwd}" if cwd else ""
        input_text = " input=<provided>" if input is not None else ""
        if self.verbose:
            print(f"[runner] START {command}{cwd_text}{input_text}", file=sys.stderr, flush=True)
        started = time.monotonic()
        stream_output = self.verbose and bool(args) and args[0] == "rsync"
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            input=input,
            text=True,
            capture_output=not stream_output,
            check=False,
        )
        elapsed = time.monotonic() - started
        result = CommandResult(
            args=tuple(args),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )
        if self.verbose:
            print(
                f"[runner] DONE {command} rc={result.returncode} elapsed={elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        if check and result.returncode != 0:
            raise CommandError(result)
        return result

    def close_ssh_masters(self) -> None:
        """Close any SSH master connections opened by this runner."""

        for host in sorted(self._masters):
            subprocess.run(
                ["ssh", *self._ssh_options(host), "-O", "exit", host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        self._masters.clear()

    def _prepare_command(self, args: Sequence[str]) -> list[str]:
        prepared = [str(arg) for arg in args]
        if not self.use_ssh_master or not prepared:
            return prepared
        if prepared[0] == "ssh" and len(prepared) >= 2:
            host = prepared[1]
            self._ensure_ssh_master(host)
            return ["ssh", *self._ssh_options(host), *prepared[1:]]
        if prepared[0] == "rsync":
            host = _rsync_remote_host(prepared)
            if host:
                self._ensure_ssh_master(host)
                return ["rsync", "-e", f"ssh {' '.join(self._ssh_options(host))}", *prepared[1:]]
        return prepared

    def _ensure_ssh_master(self, host: str) -> None:
        if host in self._masters:
            return
        self._control_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "ssh",
            *self._ssh_options(host),
            "-MNf",
            "-o",
            "ControlMaster=yes",
            host,
        ]
        if self.verbose:
            print(f"[runner] START ssh master {host}", file=sys.stderr, flush=True)
        started = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        elapsed = time.monotonic() - started
        if self.verbose:
            print(
                f"[runner] DONE ssh master {host} rc={completed.returncode} elapsed={elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        if completed.returncode != 0:
            raise CommandError(
                CommandResult(
                    args=tuple(command),
                    returncode=completed.returncode,
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                )
            )
        self._masters.add(host)

    def _ssh_options(self, host: str) -> list[str]:
        return [
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=10m",
            "-o",
            f"ControlPath={self._control_path(host)}",
        ]

    def _control_path(self, host: str) -> str:
        safe_host = "".join(char if char.isalnum() or char in "._-" else "_" for char in host)
        return str(self._control_dir / f"cm-{safe_host}")


def _rsync_remote_host(args: Sequence[str]) -> str | None:
    for arg in args[1:]:
        text = str(arg)
        if ":" not in text or text.startswith(":"):
            continue
        host, path = text.split(":", 1)
        if host and path and "/" not in host:
            return host
    return None
