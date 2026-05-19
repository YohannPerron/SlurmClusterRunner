"""Time parsing and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.args import ControlParams
from src.models import PartitionConfig


class TimeError(ValueError):
    """Raised when wall-time controls are invalid."""


@dataclass(frozen=True)
class TimeRequest:
    """Resolved wall-time settings for one job."""

    wall_time: str
    seconds: int
    use_min_time: bool


def resolve_time(controls: ControlParams, partition: PartitionConfig) -> TimeRequest:
    """Resolve requested/default wall time and validate against partition max."""

    raw_time = controls.time
    if raw_time is None:
        raw_time = partition.slurm.dev_time if controls.dev and partition.slurm.dev_time else None
    if raw_time is None:
        raw_time = partition.slurm.default_time
    if raw_time is None:
        raw_time = _hours_to_hms(partition.slurm.max_time_hours)

    seconds = parse_wall_time(str(raw_time))
    max_seconds = int(float(partition.slurm.max_time_hours) * 3600)
    if seconds > max_seconds:
        if partition.features.get("job_chaining", False):
            raise TimeError("requested TIME exceeds partition maximum; job chaining is not yet implemented")
        raise TimeError(
            f"requested TIME {format_wall_time(seconds)} exceeds partition "
            f"maximum of {partition.slurm.max_time_hours} hours"
        )

    use_min_time = (
        partition.slurm.use_min_time_default
        if controls.mintime is None
        else controls.mintime
    )
    return TimeRequest(format_wall_time(seconds), seconds, use_min_time)


def parse_wall_time(value: str) -> int:
    """Parse integer hours or ``HH:MM:SS`` into seconds."""

    if value.isdigit():
        return int(value) * 3600
    parts = value.split(":")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise TimeError("TIME must be HH:MM:SS or integer hours")
    hours, minutes, seconds = (int(part) for part in parts)
    if minutes >= 60 or seconds >= 60:
        raise TimeError("TIME minutes and seconds must be < 60")
    return hours * 3600 + minutes * 60 + seconds


def format_wall_time(seconds: int) -> str:
    """Format seconds as ``HH:MM:SS``."""

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _hours_to_hms(hours: int | float) -> str:
    return format_wall_time(int(float(hours) * 3600))
