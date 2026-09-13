# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Mockup of the Python surface of the C++ Pilot thread manager.

The design lives in cpp/solvcon/pilot/app/RThreadManager.hpp. Every body
here is a stub. WorkflowState, Error, Progress, Succeeded, Failed,
Cancelled, CancellationToken, TaskContext, WorkflowContext, WorkflowHandle,
and RThreadManager become _solvcon types. ThreadState, Task, and Workflow
stay Python base classes; the binding wraps an instance in the matching
C++ adapter when it enters a queue.
"""

import abc
import collections.abc
import dataclasses
import enum


class WorkflowState(enum.IntEnum):

    QUEUED = 0
    RUNNING = 1
    CANCELLING = 2
    FINISHED = 3


WorkflowId = int


@dataclasses.dataclass(frozen=True)
class Error:
    kind: str
    message: str


@dataclasses.dataclass(frozen=True)
class Progress:
    """A None fraction means indeterminate progress."""

    task: str
    fraction: float | None
    message: str = ''


@dataclasses.dataclass(frozen=True)
class Succeeded:
    """C++ holds the result in a PythonResult whose destructor takes the
    GIL, and hands it back unchanged on the Qt thread."""

    result: object | None = None


@dataclasses.dataclass(frozen=True)
class Failed:
    error: Error


@dataclasses.dataclass(frozen=True)
class Cancelled:
    pass


Result = Succeeded | Failed | Cancelled
ResultCallback = collections.abc.Callable[[Result], None]
StateCallback = collections.abc.Callable[[WorkflowState], None]
ProgressCallback = collections.abc.Callable[[Progress], None]


class CancellationToken:
    """Thread-safe view of a cancellation request."""

    def is_cancelled(self) -> bool:
        ...


class TaskContext:
    """Handed to Task.execute; valid until finish completes the task."""

    @property
    def workflow_id(self) -> WorkflowId:
        ...

    @property
    def cancellation(self) -> CancellationToken:
        ...

    def progress(self, fraction: float | None, message: str = '') -> None:
        ...

    def finish(self, result: Result) -> bool:
        """Return true only if this call completed the task; an accepted
        cancellation overrides result with Cancelled."""
        ...


class ThreadState(abc.ABC):
    """Long-lived objects of one task thread.

    Created, opened, closed, and used only on that thread; every task on
    the thread sees the same instance.
    """

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


ThreadStateFactory = collections.abc.Callable[[], ThreadState]


class Task(abc.ABC):
    """Executed on its task thread under the GIL; carries only
    thread-transferable data."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def execute(self, context: TaskContext, state: ThreadState) -> None:
        """Complete through context.finish. A long C++ call made here must
        release the GIL in its binding, or the Qt thread waits on it."""
        ...

    def cancel(self) -> None:
        """Runs on the task thread when cancellation is accepted while
        this task is active; must lead to context.finish."""


class WorkflowContext:
    """Valid on the workflow thread only."""

    @property
    def workflow_id(self) -> WorkflowId:
        ...

    @property
    def cancellation(self) -> CancellationToken:
        ...

    def submit(self, thread: str, task: Task,
               on_completed: ResultCallback) -> None:
        """The callback runs exactly once on the workflow thread, never
        inline on the task thread. An unregistered name completes the task
        with Failed without running it."""
        ...

    def finish(self, result: Result) -> bool:
        """Return true only if this call completed the workflow."""
        ...


class Workflow(abc.ABC):
    """Called on the workflow thread; carries only thread-transferable
    data."""

    @abc.abstractmethod
    def start(self, context: WorkflowContext) -> None:
        ...

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class WorkflowHandle:
    """Deliver workflow events to the owner on the Qt thread.

    Parented to the owner and does not own the workflow, so an event for a
    destroyed owner is dropped with the handle. Every callback runs on the
    Qt thread with the GIL held.
    """

    @property
    def workflow_id(self) -> WorkflowId:
        ...

    @property
    def state(self) -> WorkflowState:
        ...

    def cancel(self) -> bool:
        """Return true only if this call accepted cancellation; the terminal
        result is then Cancelled."""
        ...

    def on_state_changed(self, callback: StateCallback) -> None:
        ...

    def on_progress(self, callback: ProgressCallback) -> None:
        ...

    def on_finished(self, callback: ResultCallback) -> None:
        """Callback runs once with the terminal result."""
        ...


class RThreadManager:
    """Own the workflow thread and one task thread per registered name.

    Reached as RManager.instance().thread_manager. Each task thread runs
    one task at a time in FIFO order; two names never share a thread.
    Registering a name again replaces its ThreadState before the next task.
    """

    def register_thread(self, name: str,
                        factory: ThreadStateFactory | None = None) -> None:
        """Call on the Qt thread. A None factory gives an empty
        ThreadState."""
        ...

    def has_thread(self, name: str) -> bool:
        ...

    def submit(self, workflow: Workflow, *, owner) -> WorkflowHandle:
        """Owner is a non-null QObject on the Qt thread; the handle runs no
        callback before this returns."""
        ...


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
