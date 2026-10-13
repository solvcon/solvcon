# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Python surface of the C++ Pilot thread manager.

The contract lives in cpp/solvcon/pilot/app/RThreadManager.hpp.
``ThreadState``, ``Task``, and ``Workflow`` stay Python base classes; the
binding wraps an instance in the matching C++ adapter when it enters a
queue.
"""

import abc

from ._pilot_core import (
    CancellationToken, Cancelled, Error, Failed, RThreadManager, Succeeded,
    TaskContext, WorkflowContext, WorkflowHandle, WorkflowState)


class ThreadState:
    """Long-lived objects of one task thread.

    Created, opened, closed, and used only on that thread; every task on
    the thread sees the same instance, and the instance lives as long as
    the thread. A thread registered without a factory hands its tasks
    ``None`` instead.
    """

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


class Task(abc.ABC):
    """Executed on its task thread under the GIL; carries only
    thread-transferable data."""

    @abc.abstractmethod
    def execute(self, context: TaskContext,
                state: ThreadState | None) -> None:
        """Complete the task with ``context.finish`` before this method
        returns. Take ``context.cancellation`` once and poll it between
        chunks of work; a running call cannot be interrupted, so the chunk
        size sets the cancel latency. A long C++ call made here must
        release the GIL in its binding, or it blocks every other Python
        thread, including the Qt thread."""


class Workflow(abc.ABC):
    """Called on the workflow thread; carries only thread-transferable
    data."""

    @abc.abstractmethod
    def start(self, context: WorkflowContext) -> None:
        """Call ``context.finish`` before this method returns."""

    def cancel(self) -> None:
        """Runs on the workflow thread if the workflow is still open when
        an accepted ``WorkflowHandle.cancel`` reaches it. The result is
        ``Cancelled`` whatever this method does."""

    def close(self) -> None:
        pass


__all__ = [
    'CancellationToken',
    'Cancelled',
    'Error',
    'Failed',
    'RThreadManager',
    'Succeeded',
    'Task',
    'TaskContext',
    'ThreadState',
    'Workflow',
    'WorkflowContext',
    'WorkflowHandle',
    'WorkflowState',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
