# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Python surface of the C++ Pilot thread manager.

The contract lives in cpp/solvcon/pilot/app/RThreadManager.hpp. ``Workflow``
stays a Python base class; ``RThreadManager.submit`` wraps an instance in the
C++ adapter when it enters the queue.
"""

import abc

from ._pilot_core import (
    Cancelled, Error, Failed, RThreadManager, Succeeded, WorkflowContext,
    WorkflowHandle, WorkflowState)


class Workflow(abc.ABC):
    """Called on the workflow thread; carries only thread-transferable
    data."""

    @abc.abstractmethod
    def start(self, context: WorkflowContext) -> None:
        """Call ``context.finish`` before this method returns."""

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


__all__ = [
    'Cancelled',
    'Error',
    'Failed',
    'RThreadManager',
    'Succeeded',
    'Workflow',
    'WorkflowContext',
    'WorkflowHandle',
    'WorkflowState',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
