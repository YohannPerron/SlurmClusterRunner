"""Subprocess command abstraction."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
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


class SubprocessRunner:
    """Command runner backed by :mod:`subprocess`."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: str | None = None,
        input: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            input=input,
            text=True,
            capture_output=True,
            check=False,
        )
        result = CommandResult(
            args=tuple(args),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if check and result.returncode != 0:
            raise CommandError(result)
        return result
