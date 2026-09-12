# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Define the Python-first Pilot workflow and task thread API."""

import abc
import collections.abc
import dataclasses
import enum

from PySide6.QtCore import QObject, Signal


class WorkflowState(enum.IntEnum):
    """Describe the lifecycle state of a submitted workflow."""

    QUEUED = 0
    RUNNING = 1
    CANCELLING = 2
    SUCCEEDED = 3
    FAILED = 4
    CANCELLED = 5


@dataclasses.dataclass(frozen=True)
class WorkflowId:
    """Identify one workflow within a manager instance."""

    value: int


@dataclasses.dataclass(frozen=True)
class Error:
    """Carry an error category and display-safe message."""

    kind: str
    message: str


@dataclasses.dataclass(frozen=True)
class Progress:
    """Report progress for one task."""

    task: str
    fraction: float | None
    message: str = ''


@dataclasses.dataclass(frozen=True)
class Succeeded:
    """Carry the output of a successful task or workflow."""

    result: object | None = None


@dataclasses.dataclass(frozen=True)
class Failed:
    """Carry the error from a failed task or workflow."""

    error: Error


@dataclasses.dataclass(frozen=True)
class Cancelled:
    """Mark a task or workflow as cancelled."""


Result = Succeeded | Failed | Cancelled
CancelCallback = collections.abc.Callable[[], None]
ResultCallback = collections.abc.Callable[[Result], None]


class CancellationToken:
    """Provide a thread-safe view of a cancellation request."""

    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        ...


class TaskContext:
    """Control one asynchronously completed task."""

    @property
    def workflow_id(self) -> WorkflowId:
        """Return the current workflow identifier."""
        ...

    @property
    def cancellation(self) -> CancellationToken:
        """Return the current workflow's cancellation token."""
        ...

    def progress(self, fraction: float | None, message: str = '') -> None:
        """Report progress for the current task."""
        ...

    def on_cancel(self, callback: CancelCallback) -> None:
        """Run callback once on the task thread when cancelled.

        Registration after cancellation queues the callback immediately.
        Task completion disables callbacks that have not started.
        """
        ...

    def finish(self, result: Result) -> bool:
        """Return true only if this call completed the active task.

        Accepted cancellation replaces result with Cancelled.
        """
        ...


class ThreadState(abc.ABC):
    """Own the long-lived objects of one task thread.

    The manager creates, opens, and closes the state on its thread, so the
    objects it holds never cross a thread. Every task on that thread receives
    the same state.
    """

    def open(self) -> None:
        """Acquire resources on the assigned thread before the first task."""

    def close(self) -> None:
        """Release resources on the assigned thread after the last task."""


ThreadStateFactory = collections.abc.Callable[[], ThreadState]


class Task(abc.ABC):
    """Define one unit of work for the I/O or computing thread.

    A task contains only data that is safe to transfer between threads. The
    scheduler executes it and releases its reference on the assigned thread.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the stable task name used in progress reports."""
        ...

    @abc.abstractmethod
    def execute(self, context: TaskContext, state: ThreadState) -> None:
        """Start the task and later complete it through context.finish.

        The scheduler starts no other task until completion. State is the
        thread's ThreadState. Long C++ calls made here must release the Python
        GIL.
        """
        ...


class WorkflowContext:
    """Submit tasks for one workflow."""

    @property
    def workflow_id(self) -> WorkflowId:
        """Return the current workflow identifier."""
        ...

    @property
    def cancellation(self) -> CancellationToken:
        """Return the current workflow's cancellation token."""
        ...

    def io(self, task: Task, on_completed: ResultCallback) -> None:
        """Queue a task on the I/O thread and deliver its completion.

        The callback runs exactly once on the workflow thread and never runs
        inline on the I/O thread.
        """
        ...

    def compute(self, task: Task, on_completed: ResultCallback) -> None:
        """Queue a task on the computing thread and deliver its completion.

        The callback runs exactly once on the workflow thread and never runs
        inline on the computing thread.
        """
        ...

    def finish(self, result: Result) -> bool:
        """Return true only if this call completed the workflow."""
        ...


class Workflow(abc.ABC):
    """Coordinate one submission on the workflow thread.

    A workflow contains only data that is safe to transfer between threads.
    The manager calls its methods and releases its reference on the workflow
    thread.
    """

    @abc.abstractmethod
    def start(self, context: WorkflowContext) -> None:
        """Start coordination for the workflow."""
        ...

    def cancel(self) -> None:
        """Stop pending coordination on the workflow thread."""

    def close(self) -> None:
        """Release resources on the workflow thread."""


class WorkflowHandle(QObject):
    """Deliver state while its owner exists without owning the workflow.

    Accepted cancellation makes the eventual terminal state CANCELLED.
    """

    state_changed = Signal(WorkflowState)
    progress = Signal(Progress)
    finished = Signal(object)

    @property
    def workflow_id(self) -> WorkflowId:
        """Return the workflow identifier."""
        ...

    @property
    def state(self) -> WorkflowState:
        """Return the latest workflow state."""
        ...

    def cancel(self) -> bool:
        """Win cancellation against completion and report acceptance."""
        ...


class RThreadManager(QObject):
    """Own one workflow thread and permanent I/O and computing schedulers.

    This Python API is the initial implementation target. Deletion waits for
    task cancellation and thread joins when shutdown has not finished.

    Each scheduler runs one task at a time on its thread. The scheduler queues
    later tasks in FIFO order until the active task calls context.finish().
    Task execution, ThreadState creation, open(), and close() run on the
    assigned threads. A thread without a registered factory uses an empty
    ThreadState.

    Thread startup and ThreadState factory or open() errors emit
    startup_failed. Exceptions from tasks or workflow callbacks become Failed
    unless cancellation won. Shutdown and ThreadState close() errors emit
    shutdown_failed.
    """

    ready = Signal()
    startup_failed = Signal(Error)
    shutdown_failed = Signal(Error)
    stopped = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Create a manager owned by the Qt thread."""
        ...

    def set_io_state(self, factory: ThreadStateFactory) -> None:
        """Register the I/O thread's ThreadState factory before start."""
        ...

    def set_computing_state(self, factory: ThreadStateFactory) -> None:
        """Register the computing thread's ThreadState factory before start."""
        ...

    def start(self) -> None:
        """Start the workflow thread and both task schedulers."""
        ...

    def submit(self, workflow: Workflow, *, owner: QObject) -> WorkflowHandle:
        """Queue a workflow and return its owner-parented handle.

        The handle emits no signal before this method returns. Owner must be
        non-null and belong to the manager's Qt thread.
        """
        ...

    def shutdown(self) -> None:
        """Cancel workflows and stop all managed threads asynchronously."""
        ...


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
