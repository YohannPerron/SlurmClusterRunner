"""Sweep expansion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable



@dataclass(frozen=True)
class SweepAxis:
    """One user-provided sweep axis."""

    name: str
    values: list[str]
    is_hydra: bool
    position: int

    @property
    def is_variable(self) -> bool:
        """Whether this axis changes across expanded jobs."""

        return len(self.values) > 1


@dataclass(frozen=True)
class SweepJob:
    """One expanded combination of original CLI tokens.

    ``tokens`` is the canonical expanded token list. Control-parameter
    extraction runs after this step, so entries such as ``GPU=4`` appear here
    first and are consumed later by ``split_control_params``.
    """

    tokens: list[str]
    positional_args: list[str]
    hydra_overrides: list[str]
    variable_params: dict[str, str]


@dataclass(frozen=True)
class SweepPlan:
    """Parsed axes and their Cartesian expansion."""

    axes: list[SweepAxis]
    jobs: list[SweepJob]

    @property
    def variable_axes(self) -> list[SweepAxis]:
        """Axes with more than one value."""

        return [axis for axis in self.axes if axis.is_variable]


def split_bracket_aware(value: str) -> list[str]:
    """Split a comma-separated value while preserving bracketed commas.

    Brackets (), [], {} and simple/double quoted substrings are treated as
    protected regions. Empty items are preserved so invalid user input remains
    visible to later validation rather than silently disappearing.
    """

    parts: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escape = False
    pairs = {"[": "]", "(": ")", "{": "}"}

    for index, char in enumerate(value):
        if escape:
            escape = False
            continue
        if char == "\\" and quote is not None:
            escape = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if stack and char == stack[-1]:
            stack.pop()
            continue
        if char == "," and not stack:
            parts.append(value[start:index])
            start = index + 1

    parts.append(value[start:])
    return parts


def parse_sweep(tokens: Iterable[str]) -> SweepPlan:
    """Classify forwarded tokens and expand them as a Cartesian product."""

    axes: list[SweepAxis] = []
    for position, token in enumerate(tokens):
        key, sep, raw_value = token.partition("=")
        if sep:
            values = split_bracket_aware(raw_value)
            axes.append(SweepAxis(name=key, values=values, is_hydra=True, position=position))
        else:
            values = split_bracket_aware(token)
            axes.append(
                SweepAxis(
                    name=f"arg{position}",
                    values=values,
                    is_hydra=False,
                    position=position,
                )
            )

    jobs: list[SweepJob] = []
    value_product = product(*(axis.values for axis in axes)) if axes else [()]
    for combination in value_product:
        tokens: list[str] = []
        positional_args: list[str] = []
        hydra_overrides: list[str] = []
        variable_params: dict[str, str] = {}
        for axis, value in zip(axes, combination, strict=True):
            if axis.is_hydra:
                token = f"{axis.name}={value}"
                hydra_overrides.append(token)
                tokens.append(token)
            else:
                positional_args.append(value)
                tokens.append(value)
            if axis.is_variable:
                variable_params[axis.name] = value
        jobs.append(SweepJob(tokens, positional_args, hydra_overrides, variable_params))

    return SweepPlan(axes=axes, jobs=jobs)
