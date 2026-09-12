# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import dataclasses
import os
import unittest

from PySide6.QtCore import QObject
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from solvcon.benchmark import collector, matmul
from solvcon.pilot import _thread_manager
from solvcon.track import mcap

RECORDING = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'mcap', 'fake_recording.mcap')
TOPIC = '/sim/ego/drive_mode'


class RecordingState(_thread_manager.ThreadState):
    """Hold the open MCAP reader on the I/O thread across viewer tasks."""

    def __init__(self):
        self.reader = None
        self.closed = False

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.closed = True


@dataclasses.dataclass(frozen=True)
class OpenRecordingTask(_thread_manager.Task):
    name = 'open'
    path: str

    def execute(self, context, state):
        if state.reader is not None:
            state.reader.close()
        state.reader = mcap.Reader(self.path)
        context.finish(_thread_manager.Succeeded(state.reader.topics()))


@dataclasses.dataclass(frozen=True)
class ExtractTopicTask(_thread_manager.Task):
    name = 'extract'
    topic: str

    def execute(self, context, state):
        if context.cancellation.is_cancelled():
            context.finish(_thread_manager.Cancelled())
            return
        extraction = state.reader.extract(self.topic)
        context.finish(_thread_manager.Succeeded(extraction))


class OpenRecordingWorkflow(_thread_manager.Workflow):
    """Open a recording and list its topics for the viewer dock."""

    def __init__(self, path):
        self.path = path

    def start(self, context):
        context.io(OpenRecordingTask(self.path), context.finish)


class ExtractTopicWorkflow(_thread_manager.Workflow):
    """Extract one topic from the recording already open on the I/O thread."""

    def __init__(self, topic):
        self.topic = topic

    def start(self, context):
        context.io(ExtractTopicTask(self.topic), context.finish)


class CancelledByProgress(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class BenchmarkTask(_thread_manager.Task):
    """Run collector.collect and cancel between kernels through progress."""

    name = 'benchmark'
    spec: matmul.MatmulSpec

    def execute(self, context, state):
        def progress(phase, kernel):
            if context.cancellation.is_cancelled():
                raise CancelledByProgress()
            context.progress(None, f'{phase} {kernel}')

        try:
            comparison = collector.collect(self.spec, progress=progress)
        except CancelledByProgress:
            context.finish(_thread_manager.Cancelled())
            return
        context.finish(_thread_manager.Succeeded(comparison))


class BenchmarkWorkflow(_thread_manager.Workflow):
    def __init__(self, spec):
        self.spec = spec
        self.cancel_requested = False

    def start(self, context):
        context.compute(BenchmarkTask(self.spec), context.finish)

    def cancel(self):
        self.cancel_requested = True


def make_spec(shape=(2, 3), rounds=6):
    rows, inner = shape
    return matmul.MatmulSpec.from_dict({
        'operation': 'matmul',
        'lhs': {'shape': [rows, inner], 'strides': [inner, 1]},
        'rhs': {'shape': [inner, rows], 'strides': [rows, 1]},
        'dtype': 'float64',
        'sampling': {'warmups': 1, 'repetitions': 2, 'rounds': rounds},
        'kernels': ['naive', 'blas_gemm'],
    })


@unittest.skip('Thread manager API has no implementation yet')
class ThreadManagerApiDemoTC(unittest.TestCase):
    def setUp(self):
        self.app = QApplication.instance() or QApplication([])
        self.owner = QObject(self.app)
        self.manager = _thread_manager.RThreadManager(self.app)
        self.stopped = QSignalSpy(self.manager.stopped)

    def tearDown(self):
        self.manager.shutdown()
        self._assert_emitted(self.stopped)
        self.owner.deleteLater()

    def _assert_emitted(self, spy):
        self.assertTrue(spy.count() or spy.wait(2000))

    def _start(self):
        ready = QSignalSpy(self.manager.ready)
        self.manager.start()
        self._assert_emitted(ready)

    def _wait_result(self, handle):
        finished = QSignalSpy(handle.finished)
        self._assert_emitted(finished)
        self.assertEqual(finished.count(), 1)
        return finished.at(0)[0]

    def test_mcap_viewer_reuses_one_reader(self):
        states = []

        def create_state():
            states.append(RecordingState())
            return states[-1]

        self.manager.set_io_state(create_state)
        self._start()

        handle = self.manager.submit(
            OpenRecordingWorkflow(RECORDING), owner=self.owner)
        self.assertIs(handle.parent(), self.owner)
        result = self._wait_result(handle)
        self.assertIsInstance(result, _thread_manager.Succeeded)
        self.assertEqual(result.result[TOPIC], 'sim_msgs/msg/DriveMode')
        self.assertEqual(len(result.result), 4)
        self.assertFalse(handle.cancel())

        for _ in range(2):
            handle = self.manager.submit(
                ExtractTopicWorkflow(TOPIC), owner=self.owner)
            result = self._wait_result(handle)
            self.assertIsInstance(result, _thread_manager.Succeeded)
            self.assertEqual(len(result.result.time), 6)
            self.assertIn('mode', result.result.columns)

        self.assertEqual(len(states), 1)
        self.assertEqual(states[0].reader.path, RECORDING)
        self.assertFalse(states[0].closed)

        self.manager.shutdown()
        self._assert_emitted(self.stopped)
        self.assertTrue(states[0].closed)

    def test_benchmark_reports_progress_and_cancels_the_queued_one(self):
        self._start()

        first = self.manager.submit(
            BenchmarkWorkflow(make_spec((32, 32), rounds=200)),
            owner=self.owner)
        second_workflow = BenchmarkWorkflow(make_spec())
        second = self.manager.submit(second_workflow, owner=self.owner)
        progress = QSignalSpy(first.progress)

        self.assertTrue(second.cancel())
        self.assertFalse(second.cancel())
        self.assertIsInstance(self._wait_result(second),
                              _thread_manager.Cancelled)
        self.assertTrue(second_workflow.cancel_requested)

        result = self._wait_result(first)
        self.assertIsInstance(result, _thread_manager.Succeeded)
        self.assertEqual(
            [row['name'] for row in result.result['results']],
            ['naive', 'blas_gemm', 'numpy'])
        self.assertGreater(progress.count(), 0)
        report = progress.at(0)[0]
        self.assertEqual(report.task, 'benchmark')
        self.assertIsNone(report.fraction)
        self.assertEqual(report.message, 'comparison numpy')
        self.assertEqual(first.state, _thread_manager.WorkflowState.SUCCEEDED)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
