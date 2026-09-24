# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Select an operation at the request and artifact boundaries."""

import typing

from . import matmul
from . import spec


class Executor(typing.Protocol):
    """Execute a named kernel and identify its eligibility exception."""

    unavailable_error: type[Exception]

    def __call__(self, name: str) -> object: ...


@typing.runtime_checkable
class BenchmarkSpec(typing.Protocol):
    """Supply metadata and execution to shared benchmark components.

    The executor accepts a kernel name and exposes ``unavailable_error``.
    Operand preparation belongs in ``make_executor``, outside timed calls.
    """

    dtype: str
    output_shape: tuple
    kernels: tuple
    sampling: spec.Sampling

    def to_dict(self) -> dict: ...
    def make_executor(self) -> Executor: ...


def parse_spec(data):
    """Parse a supported operation's exact specification."""
    return matmul.MatmulSpec.from_dict(data)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
