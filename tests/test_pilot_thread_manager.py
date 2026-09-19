# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import functools
import os
import subprocess
import sys
import threading
import time
import unittest

import solvcon

if solvcon.HAS_PILOT:
    import shiboken6
    from PySide6.QtCore import (
        QCoreApplication, QEvent, QObject, QTimer)
    from PySide6.QtWidgets import QApplication

    from solvcon.pilot import RManager
    from solvcon.pilot._thread_manager import (
        Cancelled, Error, Failed, Succeeded, Task, Workflow, WorkflowState)
else:
    Task = Workflow = object


class ImmediateWorkflow(Workflow):

    def __init__(self, result=None):
        self.result = result
        self.thread_id = None
        self.workflow_id = None
        self.finishes = []
        self.closed = False

    def start(self, context):
        self.thread_id = threading.get_ident()
        self.workflow_id = context.workflow_id
        self.finishes.append(context.finish(Succeeded(self.result)))
        self.finishes.append(context.finish(Succeeded(self.result)))

    def close(self):
        self.closed = True


class WaitingWorkflow(Workflow):

    def __init__(self, started, gate):
        self.started = started
        self.gate = gate

    def start(self, context):
        self.started.set()
        self.gate.wait(5)
        context.finish(Succeeded('done'))


class RaisingWorkflow(Workflow):

    def __init__(self, raise_in_close=False):
        self.raise_in_close = raise_in_close

    def start(self, context):
        if not self.raise_in_close:
            raise ValueError('boom')
        context.finish(Succeeded())

    def close(self):
        if self.raise_in_close:
            raise RuntimeError('close boom')


class SlowWorkflow(Workflow):

    def start(self, context):
        time.sleep(0.05)
        context.finish(Succeeded())


class RecordingTask(Task):
    """Record the label and the thread, then finish with the label."""

    def __init__(self, label, log, set_event=None, wait_for=None,
                 raises=False, finishes=True):
        self.label = label
        self.log = log
        self.set_event = set_event
        self.wait_for = wait_for
        self.raises = raises
        self.finishes = finishes

    def execute(self, context, state):
        self.log.append((self.label, threading.get_ident()))
        if self.set_event is not None:
            self.set_event.set()
        if self.wait_for is not None:
            self.log.append((self.label, self.wait_for.wait(5)))
        if self.raises:
            raise ValueError(self.label)
        if self.finishes:
            context.finish(Succeeded(self.label))


class EarlyFinishWorkflow(Workflow):
    """Finish before the submitted task completes."""

    def __init__(self, task):
        self.task = task
        self.results = []

    def start(self, context):
        context.submit('alpha', self.task, self.results.append)
        context.finish(Succeeded('early'))


class TaskWorkflow(Workflow):
    """Submit tasks and finish with their results in completion order."""

    def __init__(self, submissions, raise_in_callback=False):
        self.submissions = submissions
        self.raise_in_callback = raise_in_callback
        self.results = []
        self.thread_ids = []

    def start(self, context):
        self.thread_ids.append(threading.get_ident())
        for thread, task in self.submissions:
            context.submit(
                thread, task, functools.partial(self.completed, context))

    def completed(self, context, result):
        self.thread_ids.append(threading.get_ident())
        self.results.append(result)
        if self.raise_in_callback:
            raise RuntimeError('callback boom')
        if len(self.results) == len(self.submissions):
            context.finish(Succeeded(list(self.results)))


@unittest.skipUnless(
    solvcon.HAS_PILOT
    and os.environ.get('SOLVCON_THREAD_SHUTDOWN_HELPER') == '1',
    'shutdown subprocess helper')
def test_thread_manager_shutdown_helper():
    app = QApplication.instance() or QApplication([])
    owner = QObject()
    for _ in range(20):
        RManager.instance.thread_manager.submit(SlowWorkflow(), owner=owner)
    app.processEvents()


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class ThreadManagerTC(unittest.TestCase):

    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.owner = QObject()
        self.manager = RManager.instance.thread_manager

    def tearDown(self):
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(
            self.owner, QEvent.Type.DeferredDelete)

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate())

    def run_workflow(self, workflow):
        results = []
        handle = self.manager.submit(workflow, owner=self.owner)
        handle.on_finished(results.append)
        self.wait_for(lambda: bool(results))
        return handle, results[0]

    def test_value_types(self):
        error = Error('ValueError', 'boom')
        self.assertEqual((error.kind, error.message), ('ValueError', 'boom'))
        self.assertEqual(Failed(error).error.kind, 'ValueError')
        value = object()
        self.assertIs(Succeeded(value).result, value)
        self.assertIsNone(Succeeded().result)
        self.assertIsInstance(Cancelled(), Cancelled)
        self.assertEqual(
            [int(s) for s in (WorkflowState.QUEUED, WorkflowState.RUNNING,
                              WorkflowState.FINISHED)], [0, 1, 3])

    def test_workflow_returns_on_qt_thread(self):
        value = object()
        workflow = ImmediateWorkflow(value)
        callback_threads = []
        results = []

        handle = self.manager.submit(workflow, owner=self.owner)
        handle.on_finished(
            lambda result: (
                callback_threads.append(threading.get_ident()),
                results.append(result)))
        self.assertEqual(results, [])

        self.wait_for(lambda: bool(results))
        self.assertNotEqual(workflow.thread_id, threading.get_ident())
        self.assertEqual(callback_threads, [threading.get_ident()])
        self.assertIs(results[0].result, value)
        self.assertEqual(workflow.finishes, [True, False])
        self.assertEqual(workflow.workflow_id, handle.workflow_id)
        self.assertTrue(workflow.closed)

    def test_states_run_in_order(self):
        states = []
        handle = self.manager.submit(ImmediateWorkflow(), owner=self.owner)
        self.assertEqual(handle.state, WorkflowState.QUEUED)
        handle.on_state_changed(states.append)
        self.wait_for(lambda: handle.state == WorkflowState.FINISHED)
        self.assertEqual(
            states, [WorkflowState.RUNNING, WorkflowState.FINISHED])
        results = []
        handle.on_finished(results.append)
        self.assertIsInstance(results[0], Succeeded)

    def test_late_callback_runs_once(self):
        results = []
        handle = self.manager.submit(
            ImmediateWorkflow('once'), owner=self.owner)
        handle.on_state_changed(
            lambda state: (
                handle.on_finished(results.append)
                if state == WorkflowState.FINISHED else None))
        self.wait_for(lambda: handle.state == WorkflowState.FINISHED)
        self.assertEqual([r.result for r in results], ['once'])

    def test_owner_destroyed_by_a_state_callback(self):
        owner = QObject()
        handle = self.manager.submit(ImmediateWorkflow(), owner=owner)
        results = []
        states = []
        handle.on_finished(results.append)
        handle.on_state_changed(
            lambda state: (
                states.append(state),
                shiboken6.delete(owner)
                if state == WorkflowState.FINISHED else None))
        self.wait_for(lambda: WorkflowState.FINISHED in states)
        self.assertEqual(results, [])

    def test_qt_timer_runs_while_workflow_waits(self):
        started = threading.Event()
        gate = threading.Event()
        handle = self.manager.submit(
            WaitingWorkflow(started, gate), owner=self.owner)
        self.assertTrue(started.wait(2))

        timer_threads = []
        QTimer.singleShot(
            0,
            lambda: (timer_threads.append(threading.get_ident()), gate.set()))
        results = []
        handle.on_finished(results.append)
        self.wait_for(lambda: bool(results))
        self.assertEqual(timer_threads, [threading.get_ident()])

    def test_start_exception_becomes_failed(self):
        _, result = self.run_workflow(RaisingWorkflow())
        self.assertIsInstance(result, Failed)
        self.assertEqual(result.error.kind, 'ValueError')
        self.assertEqual(result.error.message, 'boom')

    def test_close_exception_keeps_thread_running(self):
        _, result = self.run_workflow(RaisingWorkflow(raise_in_close=True))
        self.assertIsInstance(result, Succeeded)
        _, result = self.run_workflow(ImmediateWorkflow('next'))
        self.assertEqual(result.result, 'next')

    def test_destroyed_owner_receives_no_callback(self):
        started = threading.Event()
        gate = threading.Event()
        owner = QObject()
        handle = self.manager.submit(
            WaitingWorkflow(started, gate), owner=owner)
        results = []
        handle.on_finished(results.append)
        self.assertTrue(started.wait(2))

        owner.deleteLater()
        QCoreApplication.sendPostedEvents(owner, QEvent.Type.DeferredDelete)
        gate.set()
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertEqual(results, [])

    def test_rejects_invalid_owner(self):
        with self.assertRaises(ValueError):
            self.manager.submit(ImmediateWorkflow(), owner=None)
        with self.assertRaises(TypeError):
            self.manager.submit(ImmediateWorkflow(), owner=object())
        foreign = []
        worker = threading.Thread(target=lambda: foreign.append(QObject()))
        worker.start()
        worker.join()
        with self.assertRaises(ValueError):
            self.manager.submit(ImmediateWorkflow(), owner=foreign[0])

    def test_register_thread(self):
        self.assertFalse(self.manager.has_thread('nobody'))
        self.manager.register_thread('alpha')
        self.manager.register_thread('alpha')
        self.assertTrue(self.manager.has_thread('alpha'))

    def test_tasks_on_one_thread_run_in_order(self):
        self.manager.register_thread('alpha')
        log = []
        workflow = TaskWorkflow(
            [('alpha', RecordingTask(label, log)) for label in 'abc'])
        _, result = self.run_workflow(workflow)
        self.assertEqual([r.result for r in result.result], list('abc'))
        self.assertEqual([label for label, _ in log], list('abc'))
        self.assertEqual(len({tid for _, tid in log}), 1)
        start_thread, *callback_threads = workflow.thread_ids
        self.assertEqual(set(callback_threads), {start_thread})
        self.assertNotEqual(start_thread, threading.get_ident())
        self.assertNotEqual(start_thread, log[0][1])

    def test_tasks_on_two_threads_overlap(self):
        self.manager.register_thread('alpha')
        self.manager.register_thread('beta')
        log = []
        first, second = threading.Event(), threading.Event()
        workflow = TaskWorkflow([
            ('alpha', RecordingTask('a', log, set_event=first,
                                    wait_for=second)),
            ('beta', RecordingTask('b', log, set_event=second,
                                   wait_for=first)),
        ])
        _, result = self.run_workflow(workflow)
        self.assertEqual(
            sorted(r.result for r in result.result), ['a', 'b'])
        waits = [entry for entry in log if isinstance(entry[1], bool)]
        self.assertEqual(sorted(waits), [('a', True), ('b', True)])
        threads = {entry[1] for entry in log if not isinstance(entry[1], bool)}
        self.assertEqual(len(threads), 2)

    def test_unregistered_thread_gives_failed(self):
        log = []
        workflow = TaskWorkflow([('nobody', RecordingTask('a', log))])
        _, result = self.run_workflow(workflow)
        (failed,) = result.result
        self.assertIsInstance(failed, Failed)
        self.assertEqual(failed.error.kind, 'unregistered thread')
        self.assertEqual(log, [])

    def test_task_exception_fails_the_task_only(self):
        self.manager.register_thread('alpha')
        log = []
        workflow = TaskWorkflow([
            ('alpha', RecordingTask('a', log, raises=True)),
            ('alpha', RecordingTask('b', log)),
        ])
        _, result = self.run_workflow(workflow)
        failed, succeeded = result.result
        self.assertIsInstance(failed, Failed)
        self.assertEqual(
            (failed.error.kind, failed.error.message), ('ValueError', 'a'))
        self.assertEqual(succeeded.result, 'b')

    def test_task_returning_without_a_result_fails(self):
        self.manager.register_thread('alpha')
        workflow = TaskWorkflow(
            [('alpha', RecordingTask('a', [], finishes=False))])
        _, result = self.run_workflow(workflow)
        (failed,) = result.result
        self.assertEqual(failed.error.kind, 'incomplete')

    def test_callback_exception_fails_the_workflow(self):
        self.manager.register_thread('alpha')
        workflow = TaskWorkflow(
            [('alpha', RecordingTask('a', []))], raise_in_callback=True)
        _, result = self.run_workflow(workflow)
        self.assertIsInstance(result, Failed)
        self.assertEqual(
            (result.error.kind, result.error.message),
            ('RuntimeError', 'callback boom'))

    def test_early_finish_still_delivers_the_task_callback(self):
        """A workflow that finishes first must not drop the promised
        completion callback of a task that is still running."""
        self.manager.register_thread('alpha')
        log = []
        workflow = EarlyFinishWorkflow(RecordingTask('a', log))
        _, result = self.run_workflow(workflow)
        self.assertEqual(result.result, 'early')
        self.wait_for(lambda: bool(workflow.results))
        self.assertEqual([r.result for r in workflow.results], ['a'])

    def test_process_exits_with_queued_work(self):
        if os.path.basename(sys.executable).lower() not in (
                'pilot', 'pilot.exe'):
            self.skipTest('requires the Pilot executable')
        env = os.environ.copy()
        env['SOLVCON_THREAD_SHUTDOWN_HELPER'] = '1'
        env['PYTEST_OPTS'] = '-q -k test_thread_manager_shutdown_helper'
        completed = subprocess.run(
            [sys.executable, '--mode=pytest'],
            capture_output=True,
            env=env,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
