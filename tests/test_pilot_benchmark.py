# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Exercise benchmark widgets without mapping a top-level window."""

import contextlib
import dataclasses
import json
import os
import pathlib
import tempfile
import time
import unittest
import unittest.mock

from solvcon import system
from solvcon.benchmark import matmul, results, spec

try:
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
    import shiboken6
except ImportError:
    QtWidgets = None
else:
    from solvcon.pilot.benchmark import _inspector
    from solvcon.pilot.benchmark import _run


@contextlib.contextmanager
def mock_process(control):
    """Replace process I/O while preserving the widget's Qt connections."""
    process = control._process
    methods = ('start', 'kill', 'write', 'closeWriteChannel', 'canReadLine',
               'readLine', 'readAllStandardOutput', 'readAllStandardError')
    patches = dict.fromkeys(methods, unittest.mock.DEFAULT)
    with unittest.mock.patch.multiple(process, **patches):
        process.canReadLine.return_value = False
        process.readAllStandardOutput.return_value = b''
        process.readAllStandardError.return_value = b''
        yield process


def feed_stdout(process, payload):
    """Deliver bytes through the connected stdout handler."""
    stream = QtCore.QBuffer()
    stream.setData(payload)
    stream.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
    process.canReadLine.side_effect = stream.canReadLine
    process.readLine.side_effect = stream.readLine
    process.readyReadStandardOutput.emit()


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class RunPanelTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        self.control = _run.RunPanel()
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name) / 'result.json'
        self.spec = matmul.MatmulSpec(
            lhs=spec.OperandSpec((2, 3), (3, 1)),
            rhs=spec.OperandSpec((3, 2), (2, 1)), dtype='float64',
            sampling=spec.Sampling(1, 2, 2), kernels=('naive',))
        self.events = []
        self.control.completed.connect(
            lambda path: self.events.append(('result', path)))
        self.control.failed.connect(
            lambda message: self.events.append(('error', message)))
        self.control.stopped.connect(
            lambda: self.events.append(('stopped', None)))

    def tearDown(self):
        process = self.control._process
        stopped = QtCore.QProcess.ProcessState.NotRunning
        self.control.stop()
        self.wait_for(lambda: process.state() == stopped)
        self.control.close()
        self.control.deleteLater()
        self.app.sendPostedEvents(self.control, QtCore.QEvent.DeferredDelete)
        self.directory.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate(), self.control.status.text())

    def start_idle_process(self):
        command = system.python_command(__file__)
        with unittest.mock.patch.object(system, 'python_command',
                                        return_value=command):
            self.control.start(self.spec, self.path)

    @contextlib.contextmanager
    def mock_worker(self):
        """Initialize a run whose process signals are driven by the test."""
        with mock_process(self.control) as process:
            self.control.start(self.spec, self.path)
            yield process

    def assert_filters_events(self, active):
        receiver = QtCore.QObject()
        event = QtCore.QEvent(QtCore.QEvent.Type.User)
        with unittest.mock.patch.object(self.control, 'eventFilter',
                                        return_value=False) as filtered:
            self.app.sendEvent(receiver, event)
        self.assertEqual(filtered.call_count, int(active))

    def assert_finished(self, kind):
        self.wait_for(lambda: not self.control.running)
        self.assert_filters_events(False)
        self.assertEqual([event[0] for event in self.events], [kind])
        self.assertEqual(self.control._process.state(),
                         QtCore.QProcess.ProcessState.NotRunning)
        self.assertFalse(self.control.stop_button.isEnabled())
        self.assertFalse(self.control._timer.isActive())
        success = kind == 'result'
        self.assertEqual(self.control.progress.value(), int(success))
        self.assertEqual(self.control.progress.isTextVisible(), success)
        self.assertTrue(self.control.progress.isHidden())

    def assert_failed(self, message):
        self.assert_finished('error')
        self.assertEqual(self.events[0][1], message)

    def assert_reusable(self):
        self.events.clear()
        with self.mock_worker() as process:
            with self.assertRaises(RuntimeError) as caught:
                self.control.start(self.spec, self.path)
            self.assertEqual(str(caught.exception),
                             'a benchmark is already running')
            process.started.emit()
            request = json.loads(process.write.call_args.args[0])
            self.assertEqual(request['spec'], self.spec.to_dict())
            self.assertEqual(request['output_path'], str(self.path))
            process.closeWriteChannel.assert_called_once()
            self.control._handle_event({
                'type': 'result', 'artifact_path': str(self.path),
            })
            process.finished.emit(0, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_finished('result')

    def test_run_and_repeat(self):
        for _ in range(2):
            self.assert_reusable()

    def test_progress_stop_and_recover(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        with self.mock_worker() as process:
            self.control._handle_event(event)
            self.assertEqual(self.control.status.text(), 'Timing: naive')
            self.assertEqual(self.events, [])
            self.assertEqual(self.control.progress.value(), 16)
            self.assertEqual(self.control.progress.text(), '16% (1/6 units)')
            self.control.stop_button.click()
            process.kill.assert_called_once()
            self.assertTrue(self.control.running)
            self.assertEqual(self.events, [])
            process.finished.emit(-1, QtCore.QProcess.ExitStatus.CrashExit)
        self.assert_finished('stopped')
        self.assert_reusable()

    def test_waits_for_complete_event(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        payload = json.dumps(event).encode()
        chunks = (
            (payload[:-1], 'Preparing'),
            (payload[-1:], 'Preparing'),
            (b'\n', 'Timing: naive'),
        )
        stream = QtCore.QBuffer()
        stream.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)
        self.addCleanup(stream.close)
        with self.mock_worker() as process:
            process.canReadLine.side_effect = stream.canReadLine
            process.readLine.side_effect = stream.readLine
            for chunk, expected in chunks:
                unread = stream.pos()
                stream.seek(stream.size())
                stream.write(chunk)
                stream.seek(unread)
                process.readyReadStandardOutput.emit()
                self.assertEqual(self.control.status.text(), expected)
                self.assertEqual(self.control._error, '')
                self.assertEqual(self.events, [])
            self.assertTrue(stream.atEnd())
            self.assertEqual(self.control.progress.text(), '16% (1/6 units)')

    def test_elapsed_while_running(self):
        with (
            self.mock_worker(),
            unittest.mock.patch.object(self.control, '_clock') as clock,
        ):
            self.assertFalse(self.control.progress.isHidden())
            self.assertTrue(self.control._timer.isActive())
            self.assertEqual(self.control._timer.interval(), 100)
            clock.elapsed.return_value = 100
            self.control._timer.timeout.emit()
            self.assertEqual(self.control.elapsed.text(), 'Elapsed: 0.1 s')
            self.assertTrue(self.control.running)
            self.assertEqual(self.events, [])

    def test_elapsed_duration(self):
        cases = ((59900, '59.9 s'), (60000, '00:01:00'),
                 (3599000, '00:59:59'), (3600000, '01:00:00'),
                 (90061200, '25:01:01'))
        with unittest.mock.patch.object(self.control, '_clock') as clock:
            for milliseconds, text in cases:
                with self.subTest(milliseconds=milliseconds):
                    clock.elapsed.return_value = milliseconds
                    self.control._update_elapsed()
                    self.assertEqual(self.control.elapsed.text(),
                                     f'Elapsed: {text}')

    def test_large_progress_counts(self):
        with self.mock_worker():
            self.assertFalse(self.control.progress.isTextVisible())
            total = 2**33
            cases = (
                ('warmup', 0, 0),
                ('timing', total // 2, 50),
                ('timing', total, 100),
            )
            for phase, completed, percent in cases:
                with self.subTest(phase=phase, completed=completed):
                    self.control._handle_event({
                        'type': 'progress', 'phase': phase, 'kernel': 'numpy',
                        'completed': completed, 'total': total,
                    })
                    self.assertEqual(self.control.progress.value(), percent)
                    self.assertTrue(self.control.progress.isTextVisible())
            self.assertEqual(self.events, [])
            self.assertTrue(self.control.running)

    def test_finishing_hides_progress(self):
        with self.mock_worker():
            self.control._handle_event({
                'type': 'progress', 'phase': 'timing', 'kernel': 'numpy',
                'completed': 6, 'total': 6,
            })
            self.assertTrue(self.control.progress.isTextVisible())
            self.assertEqual(self.control.progress.value(), 100)
            self.control._handle_event({
                'type': 'progress', 'phase': 'finishing', 'kernel': None,
            })
            self.assertEqual(self.control.progress.maximum(), 0)
            self.assertFalse(self.control.progress.isTextVisible())
            self.assertEqual(self.control.status.text(), 'Finishing')
            self.assertEqual(self.events, [])
            self.assertTrue(self.control.running)

    def test_rejects_invalid_progress(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        counts_error = 'invalid worker progress counts'
        progress_error = 'invalid worker progress'
        cases = (
            ('missing_count', {'completed': None}, counts_error),
            ('boolean_count', {'completed': True}, counts_error),
            ('negative_count', {'completed': -1}, counts_error),
            ('exceeds_total', {'completed': 7}, counts_error),
            ('zero_total', {'total': 0}, counts_error),
            ('noninteger_total', {'total': 6.0}, counts_error),
            ('unknown_kernel', {'kernel': 'missing'}, progress_error),
            ('finishing_with_counts', {'phase': 'finishing', 'kernel': None},
             progress_error),
        )
        with self.mock_worker():
            for name, change, message in cases:
                with self.subTest(case=name):
                    with self.assertRaises(ValueError) as caught:
                        self.control._handle_event({**event, **change})
                    self.assertEqual(str(caught.exception), message)
            self.assertEqual(self.events, [])
            self.assertEqual(self.control.status.text(), 'Preparing')

    def test_rejects_inconsistent_progress(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        cases = (
            ('regressing_count', {'completed': 0}),
            ('changing_total', {'total': 7}),
        )
        with self.mock_worker():
            values = []
            self.control.progress.valueChanged.connect(values.append)
            self.control._handle_event(event)
            for name, change in cases:
                with self.subTest(case=name):
                    with self.assertRaises(ValueError) as caught:
                        self.control._handle_event({**event, **change})
                    self.assertEqual(str(caught.exception),
                                     'invalid worker progress counts')
            self.assertEqual(values, [16])
            self.assertEqual(self.events, [])

    def test_progress_error_recovery(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        with self.mock_worker() as process:
            events = (event, {**event, 'completed': 0})
            payload = ''.join(json.dumps(item) + '\n' for item in events)
            feed_stdout(process, payload.encode())
            process.kill.assert_called_once()
            self.assertEqual(self.events, [])
            process.finished.emit(-1, QtCore.QProcess.ExitStatus.CrashExit)
        message = 'Worker protocol error: invalid worker progress counts'
        self.assert_failed(message)
        self.assert_reusable()

    def test_thread_isolation(self):
        names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
        inherited = dict.fromkeys(names, '2')
        for threads, expected in ((3, '3'), (None, '2')):
            with (
                self.subTest(threads=threads),
                unittest.mock.patch.dict(os.environ, inherited),
                mock_process(self.control) as process,
            ):
                self.control.start(self.spec, self.path, threads=threads)
                env = self.control._process.processEnvironment()
                actual = {name: env.value(name) for name in names}
                self.assertEqual(actual, dict.fromkeys(names, expected))
                parent = {name: os.environ[name] for name in names}
                self.assertEqual(parent, inherited)
                process.finished.emit(0, QtCore.QProcess.ExitStatus.NormalExit)

    def test_invalid_threads(self):
        for threads in (0, -1, True, 1.5, '2'):
            with self.subTest(threads=threads):
                with self.assertRaises(ValueError) as caught:
                    self.control.start(self.spec, self.path, threads=threads)
                self.assertEqual(str(caught.exception),
                                 'threads must be a positive integer')
                self.assertFalse(self.control.running)

    def test_protocol_errors(self):
        json_error = ('Worker protocol error: '
                      'Expecting value: line 1 column 1 (char 0)')
        cases = (
            (b'not json\n', json_error),
            (b'{}\n', 'Worker protocol error: unknown worker event'),
            (b'{"type": "error", "message": "bad spec"}\n', 'bad spec'),
        )
        for output, message in cases:
            with self.subTest(message=message), self.mock_worker() as process:
                self.events.clear()
                feed_stdout(process, output)
                process.kill.assert_called_once()
                self.assertEqual(self.control._error, message)
                self.assertEqual(self.events, [])
                process.finished.emit(-1, QtCore.QProcess.ExitStatus.CrashExit)
                self.assert_failed(message)

    def test_protocol_error_recovery(self):
        with self.mock_worker() as process:
            feed_stdout(process, b'not json\n')
            process.kill.assert_called_once()
            self.assertEqual(self.events, [])
            process.finished.emit(-1, QtCore.QProcess.ExitStatus.CrashExit)
        message = ('Worker protocol error: '
                   'Expecting value: line 1 column 1 (char 0)')
        self.assert_failed(message)
        self.assert_reusable()

    def test_incomplete_event_on_exit(self):
        with self.mock_worker() as process:
            process.readAllStandardOutput.return_value = b'{}'
            process.finished.emit(0, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_failed('Worker protocol error: incomplete event')

    def test_exit_without_result(self):
        with self.mock_worker() as process:
            process.finished.emit(0, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_failed('Worker exited without a result')

    def test_native_error(self):
        with self.mock_worker() as process:
            process.readAllStandardError.return_value = b'native crash'
            process.readyReadStandardError.emit()
            process.readAllStandardError.return_value = b''
            process.finished.emit(3, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_failed('Worker exited with code 3\nnative crash')

    def test_result_before_nonzero_exit(self):
        with self.mock_worker() as process:
            self.control._handle_event({
                'type': 'result', 'artifact_path': 'fake',
            })
            self.assertEqual(self.events, [])
            process.finished.emit(3, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_failed('Worker exited with code 3')

    def test_crash(self):
        process = self.control._process
        running = QtCore.QProcess.ProcessState.Running
        self.start_idle_process()
        self.wait_for(lambda: process.state() == running)
        process.kill()
        self.assert_finished('error')

    def restart_after_failure(self, message):
        # Reentering QProcess inside its error signal can block Qt.
        if not self.process_errors:
            return
        self.control.failed.disconnect(self.restart_after_failure)
        self.events.clear()
        self.control.start(self.spec, self.path)

    def test_failed_start_recovers_from_signal(self):
        self.process_errors = []
        self.control._process.errorOccurred.connect(self.process_errors.append)
        self.control.failed.connect(self.restart_after_failure)
        with unittest.mock.patch.object(system, 'python_command',
                                        return_value=['/missing/worker']):
            self.control.start(self.spec, self.path)
        self.assert_finished('result')
        self.assertEqual(self.process_errors,
                         [QtCore.QProcess.ProcessError.FailedToStart])
        self.assertEqual(results.load_artifact(self.path).spec.to_dict(),
                         self.spec.to_dict())

    def test_stop_during_start_and_close(self):
        for action in (self.control.stop, self.control.close):
            self.events.clear()
            self.start_idle_process()
            action()
            self.assert_finished('stopped')

    def test_filters_application_events_only_while_running(self):
        self.assert_filters_events(False)
        with unittest.mock.patch.object(self.control, '_process') as process:
            process.canReadLine.return_value = False
            process.readAllStandardOutput.return_value = b''
            process.readAllStandardError.return_value = b''
            self.control.start(self.spec, self.path)
            self.assert_filters_events(True)
            self.control._finish(0, QtCore.QProcess.ExitStatus.NormalExit)
        self.assert_filters_events(False)

    def test_filter_ignores_layout_item_receivers(self):
        self.start_worker(WorkerStub())
        self.wait_for(lambda: self.control._process.state() ==
                      QtCore.QProcess.ProcessState.Running)
        parent = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(parent)
        widget = QtWidgets.QWidget(parent)
        layout.addWidget(widget)
        receiver = layout.itemAt(0)
        try:
            widget.deleteLater()
            self.app.sendPostedEvents(
                widget, QtCore.QEvent.Type.DeferredDelete)
            self.assertEqual(layout.count(), 0)
            # Inject the deleted item's wrapper without relying on reuse.
            for invalidated in (False, True):
                if invalidated:
                    shiboken6.invalidate(receiver)
                for kind in (QtCore.QEvent.Type.User, QtCore.QEvent.Type.Quit):
                    with self.subTest(invalidated=invalidated, kind=kind):
                        event = QtCore.QEvent(kind)
                        self.assertFalse(
                            self.control.eventFilter(receiver, event))
                        self.assertTrue(self.control.running)
                        self.assertEqual(self.events, [])
        finally:
            shiboken6.delete(parent)

    def test_quit_for_another_receiver_keeps_worker_running(self):
        self.start_worker(WorkerStub())
        self.wait_for(lambda: self.control._process.state() ==
                      QtCore.QProcess.ProcessState.Running)
        receiver = QtCore.QObject()
        self.app.sendEvent(receiver, QtCore.QEvent(QtCore.QEvent.Type.Quit))
        self.assertTrue(self.control.running)
        self.assertEqual(self.events, [])

    def test_application_quit_stops_worker_before_returning(self):
        self.start_worker(WorkerStub())
        self.wait_for(lambda: self.control._process.state() ==
                      QtCore.QProcess.ProcessState.Running)
        event = QtCore.QEvent(QtCore.QEvent.Type.Quit)
        self.assertFalse(self.control.eventFilter(self.app, event))
        self.assertFalse(self.control.running)
        self.assert_finished('stopped')


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class BenchmarkInspectorTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        self.widget = _inspector.BenchmarkInspector()
        self.path = self.widget._path
        self.fields = {**self.widget.fields, **self.widget.form.fields}
        self.set_fields(lhs_shape='2, 3', lhs_strides='-3, 1',
                        rhs_shape='3, 2', rhs_strides='0, 1',
                        warmups='0', repetitions='1', rounds='1')
        for name, box in self.widget.form.kernels.items():
            box.setChecked(name == 'naive')

    def tearDown(self):
        process = self.widget.control._process
        self.widget.control.stop()
        process.waitForFinished(15000)
        self.assertEqual(process.state(),
                         QtCore.QProcess.ProcessState.NotRunning)
        self.widget.close()
        self.widget.deleteLater()
        self.app.sendPostedEvents(self.widget, QtCore.QEvent.DeferredDelete)

    def set_fields(self, **values):
        for name, value in values.items():
            self.fields[name].setText(value)

    def write_result(self):
        request = self.widget.make_spec()
        names = request.kernels + ('numpy',)
        entries = [
            results.KernelResult(
                name, 'measured', max_abs_diff=0.0, relative_diff=0.0,
                round_elapsed_ns=[100] * request.sampling.rounds)
            for name in names
        ]
        result = results.RunResult(
            request, [list(names)] * request.sampling.rounds, entries)
        results.write_artifact(result, self.path)
        return self.path

    def complete_run(self):
        event = {'type': 'result', 'artifact_path': str(self.write_result())}
        process = self.widget.control._process
        feed_stdout(process, (json.dumps(event) + '\n').encode())
        process.finished.emit(0, QtCore.QProcess.ExitStatus.NormalExit)

    def test_spec(self):
        self.set_fields(lhs_shape='2, 4, 3', lhs_strides='20, -5, 0',
                        rhs_shape='1, 3, 2', rhs_strides='0, 4, 1',
                        warmups='0', repetitions=str(2**32), rounds='7')
        self.widget.form.dtype.setCurrentText('complex64')
        self.widget.form.kernels['blas_gemm'].setChecked(True)
        self.assertEqual(self.widget.operation.count(), 1)
        self.assertEqual(self.widget.operation.currentText(), 'Matmul')
        self.assertEqual(self.widget.make_spec().to_dict(), {
            'operation': 'matmul', 'dtype': 'complex64',
            'lhs': {'shape': [2, 4, 3], 'strides': [20, -5, 0]},
            'rhs': {'shape': [1, 3, 2], 'strides': [0, 4, 1]},
            'sampling': {'warmups': 0, 'repetitions': 2**32, 'rounds': 7},
            'kernels': ['naive', 'blas_gemm']})

    def test_form_is_ready_before_kernel_placement(self):
        inputs = QtWidgets.QWidget()
        self.addCleanup(inputs.deleteLater)
        layout = QtWidgets.QFormLayout(inputs)
        form = _inspector.MatmulForm(layout)
        sampling = spec.Sampling(0, 1, 1)

        request = form.make_spec(sampling)
        self.assertIn('naive', request.kernels)
        self.assertNotIn('blas_dot', request.kernels)
        form.add_kernels(layout)
        self.assertEqual(form.make_spec(sampling), request)

    def test_uncapped_shapes(self):
        for extent, stride in ((0, 0), (2**32, -1)):
            with self.subTest(extent=extent, stride=stride):
                self.set_fields(lhs_shape=f'{extent}, 3',
                                lhs_strides=f'{stride}, 0')
                result = self.widget.make_spec()
                self.assertEqual(result.lhs.shape, (extent, 3))
                self.assertEqual(result.lhs.strides, (stride, 0))

    def test_sampling_help(self):
        self.set_fields(warmups='3', repetitions='7', rounds='2')
        expected = '3 warmups; 2 rounds x 7 calls per kernel.'
        self.assertEqual(self.widget.sampling_help.text(), expected)
        self.set_fields(rounds='0')
        self.assertEqual(self.widget.sampling_help.text(),
                         'sampling.rounds must be at least 1')
        self.set_fields(rounds='2')
        self.assertEqual(self.widget.sampling_help.text(), expected)

    def test_kernel_recovery(self):
        self.set_fields(lhs_shape='2, 4', lhs_strides='4, 1',
                        rhs_shape='4, 2', rhs_strides='2, 1')
        boxes = self.widget.form.kernels
        winograd = boxes['winograd']
        if not winograd.isEnabled():
            self.skipTest('BLAS backend is not available')
        winograd.setChecked(True)
        boxes['blas_gemm'].setChecked(False)
        self.set_fields(lhs_shape='3, 4')
        self.assertFalse(winograd.isEnabled())
        self.assertFalse(winograd.isChecked())
        self.assertIn('even', winograd.toolTip())
        self.assertNotIn('winograd', self.widget.make_spec().kernels)
        self.set_fields(lhs_shape='2, 4')
        self.assertTrue(winograd.isEnabled())
        self.assertTrue(winograd.isChecked())
        self.assertNotIn('Unavailable', winograd.toolTip())
        self.assertFalse(boxes['blas_gemm'].isChecked())
        self.set_fields(lhs_strides='4')
        self.assertTrue(all(not box.isEnabled() for box in boxes.values()))
        self.set_fields(lhs_strides='-4, 0')
        self.assertTrue(winograd.isChecked())
        self.assertFalse(boxes['blas_gemm'].isChecked())
        self.widget.form.dtype.setCurrentText('complex64')
        self.assertTrue(winograd.isChecked())
        with mock_process(self.widget.control):
            self.widget.run_button.click()
            self.complete_run()
        self.assertTrue(winograd.isEnabled())
        self.assertFalse(boxes['blas_dot'].isEnabled())
        self.assertEqual(self.widget.control.status.text(), 'Completed')

    def test_kernel_rank_and_invalid_input(self):
        self.set_fields(lhs_shape='4', lhs_strides='-1',
                        rhs_shape='4', rhs_strides='0')
        boxes = self.widget.form.kernels
        self.assertFalse(boxes['blas_gemm'].isEnabled())
        if boxes['blas_dot'].isEnabled():
            boxes['blas_dot'].setChecked(True)
            self.assertIn('blas_dot', self.widget.make_spec().kernels)
        self.set_fields(rhs_shape='5')
        self.assertTrue(all(not box.isEnabled() for box in boxes.values()))
        self.assertIn('contraction', boxes['naive'].toolTip())
        self.set_fields(rhs_shape='4')
        self.assertTrue(boxes['naive'].isEnabled())
        self.set_fields(rounds='0', repetitions='invalid')
        for box in boxes.values():
            box.setChecked(False)
        self.set_fields(rhs_shape='5')
        self.assertTrue(all(not box.isEnabled() for box in boxes.values()))
        self.set_fields(rhs_shape='4')
        self.assertTrue(boxes['naive'].isEnabled())
        self.assertTrue(all(not box.isChecked() for box in boxes.values()))

    def test_reject_input(self):
        cases = (
            ('lhs_shape', '2,,3', 'lhs shape: enter comma-separated integers'),
            ('lhs_shape', '2, 4',
             'matmul contraction dimensions do not match'),
            ('rounds', '1, 2', 'rounds: enter one integer'),
            ('threads', '0', 'threads must be a positive integer'))
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                previous = self.fields[field].text()
                self.fields[field].setText(value)
                self.widget.run_button.click()
                self.assertEqual(self.widget.error.text(), error)
                self.assertFalse(self.widget.control.running)
                self.assertFalse(self.path.exists())
                self.assertTrue(self.widget.run_button.isEnabled())
                self.fields[field].setText(previous)
        self.widget.form.kernels['naive'].setChecked(False)
        self.widget.run_button.click()
        self.assertEqual(self.widget.error.text(), 'kernels must not be empty')
        self.assertFalse(self.widget.control.running)

    def test_run_and_repeat(self):
        for rounds in ('1', '2'):
            self.fields['rounds'].setText(rounds)
            expected = self.widget.make_spec().to_dict()
            with mock_process(self.widget.control) as process:
                self.widget.run_button.click()
                process.started.emit()
                request = json.loads(process.write.call_args.args[0])
                self.assertEqual(request['spec'], expected)
                self.assertEqual(request['output_path'], str(self.path))
                self.assertTrue(self.widget.control.running)
                self.assertFalse(self.widget.inputs.isEnabled())
                self.assertFalse(self.widget.run_button.isEnabled())
                self.assertFalse(self.widget.save_button.isEnabled())
                self.complete_run()
            self.assertEqual(self.widget.control.status.text(), 'Completed')
            self.assertEqual(results.load_artifact(self.path).spec.to_dict(),
                             expected)
            self.assertTrue(self.widget.inputs.isEnabled())
            self.assertTrue(self.widget.run_button.isEnabled())
            self.assertTrue(self.widget.save_button.isEnabled())
            table = self.widget.results.table
            self.assertEqual(table.rowCount(), 2)
            self.assertEqual(table.item(0, 0).text(), 'naive')
            self.assertEqual(table.item(1, 0).text(), 'NumPy')

    def test_result_uses_artifact(self):
        self.widget.control.completed.emit(str(self.write_result()))
        snapshot = self.widget.results.summary.text()
        self.set_fields(lhs_shape='4, 3', lhs_strides='3, 1')
        self.widget.control.completed.emit(str(self.path))
        self.assertEqual(self.widget.results.summary.text(), snapshot)
        self.assertIn('A (2, 3) strides (-3, 1)', snapshot)

    def test_invalid_result_disables_save(self):
        self.widget.control.completed.emit(str(self.write_result()))
        self.assertTrue(self.widget.save_button.isEnabled())
        snapshot = self.widget.results.summary.text()
        table = self.widget.results.table
        median = table.item(0, 1).text()
        self.path.write_text('{}', encoding='utf8')
        self.widget.control.completed.emit(str(self.path))
        self.assertFalse(self.widget.save_button.isEnabled())
        message = ("artifact is missing fields: "
                   "['results', 'round_orders', 'spec']")
        self.assertEqual(self.widget.error.text(), message)
        self.assertEqual(self.widget.results.summary.text(), snapshot)
        self.assertEqual(table.item(0, 1).text(), median)

    def test_stop_and_close_recovery(self):
        control = self.widget.control
        actions = (control.stop_button.click, self.widget.close)
        for action in actions:
            with (
                self.subTest(action=action.__name__),
                mock_process(control) as process,
            ):
                self.widget.run_button.click()
                action()
                process.kill.assert_called_once()
                self.assertFalse(self.widget.run_button.isEnabled())
                process.finished.emit(-1, QtCore.QProcess.ExitStatus.CrashExit)
                self.assertEqual(control.status.text(), 'Stopped')
                self.assertTrue(self.widget.inputs.isEnabled())
                self.assertTrue(self.widget.run_button.isEnabled())
                self.widget.run_button.click()
                self.complete_run()
                self.assertEqual(control.status.text(), 'Completed')

    def test_failure_recovery(self):
        control = self.widget.control
        with mock_process(control) as process:
            self.widget.run_button.click()
            process.finished.emit(1, QtCore.QProcess.ExitStatus.NormalExit)
            self.assertEqual(control.status.text(),
                             'Failed: Worker exited with code 1')
            self.assertTrue(self.widget.inputs.isEnabled())
            self.assertTrue(self.widget.run_button.isEnabled())
            self.widget.run_button.click()
            self.complete_run()
        self.assertEqual(control.status.text(), 'Completed')

    def test_save(self):
        self.widget.control.completed.emit(str(self.write_result()))
        path = self.path.with_name('saved.json')
        with unittest.mock.patch.object(
                QtWidgets.QFileDialog, 'getSaveFileName') as dialog:
            dialog.return_value = ('', '')
            with unittest.mock.patch.object(
                    results, 'write_artifact') as write:
                self.widget.save_button.click()
                write.assert_not_called()
            dialog.return_value = (str(path), '')
            with unittest.mock.patch.object(
                    results, 'write_artifact', side_effect=OSError('Full')):
                self.widget.save_button.click()
            self.assertEqual(self.widget.error.text(), 'Full')
            self.fields['rounds'].setText('7')
            self.widget.save_button.click()
        self.assertEqual(self.widget.error.text(), '')
        self.assertEqual(results.load_artifact(path),
                         results.load_artifact(self.path))


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class ResultViewTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        # Exercise timing headers wider than Qt's default column width.
        font = self.app.font()
        self.addCleanup(self.app.setFont, font)
        font = QtGui.QFont(font)
        font.setPointSize(14)
        self.app.setFont(font)
        self.widget = _inspector.ResultView(_inspector.MatmulForm.describe)
        self.addCleanup(self.app.sendPostedEvents,
                        self.widget, QtCore.QEvent.DeferredDelete)
        self.addCleanup(self.widget.deleteLater)
        operand = spec.OperandSpec((2, 2), (2, 1))
        request = matmul.MatmulSpec(
            lhs=operand, rhs=operand, dtype='float64',
            sampling=spec.Sampling(2, 4, 3),
            kernels=('naive', 'blas_dot', 'winograd'))
        kernels = [
            results.KernelResult(
                'naive', 'measured', max_abs_diff=0.25, relative_diff=0.125,
                round_elapsed_ns=[400, 800, 1200]),
            results.KernelResult('blas_dot', 'ineligible',
                                 reason='Vectors only'),
            results.KernelResult('winograd', 'invalid', reason='Nonfinite'),
            results.KernelResult(
                'numpy', 'measured', max_abs_diff=0.0, relative_diff=0.0,
                round_elapsed_ns=[800, 1600, 2400]),
        ]
        self.result = results.RunResult(
            spec=request, round_orders=[['numpy', 'naive']] * 3,
            results=kernels)

    def row_text(self, row):
        table = self.widget.table
        return [table.item(row, col).text()
                for col in range(table.columnCount())]

    def hover_chart(self):
        """Hover within the first of the fixture's two measured rows."""
        chart = self.widget.chart
        chart.grab()
        point = QtCore.QPointF(chart.width() / 2, chart.height() / 4)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove, point, point,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(chart, event)

    def test_summary_and_statuses(self):
        self.widget.set_result(self.result)
        self.assertIn('float64; A (2, 2) strides (2, 1)',
                      self.widget.summary.text())
        self.assertEqual(self.row_text(0),
                         ['naive', '200', '290', '0.25', '0.125', 'measured'])
        self.assertEqual(self.row_text(1),
                         ['blas_dot', '-', '-', '-', '-', 'ineligible'])
        self.assertEqual(self.row_text(2),
                         ['winograd', '-', '-', '-', '-', 'invalid'])
        self.assertEqual(self.row_text(3),
                         ['NumPy', '400', '580', '0', '0', 'measured'])
        self.assertIn('3 round averages; 4 calls/round',
                      self.widget.summary.text())
        self.assertIn('Vectors only', self.widget.table.item(1, 5).toolTip())
        self.assertIn('Nonfinite', self.widget.table.item(2, 5).toolTip())

    def test_time_units(self):
        original = [entry.round_elapsed_ns.copy()
                    for entry in self.result.results]
        for scale, unit in ((1, 'ns'), (1000, '\u00b5s'),
                            (10**6, 'ms'), (10**9, 's')):
            with self.subTest(unit=unit):
                for entry, samples in zip(self.result.results, original):
                    entry.round_elapsed_ns = [ns * scale for ns in samples]
                self.widget.set_result(self.result)
                self.assertEqual(self.row_text(0)[1:3], ['200', '290'])
                header = self.widget.table.horizontalHeaderItem(1)
                self.assertEqual(header.text(), f'Median ({unit}/call)')
                self.assertEqual(self.widget.chart._unit, unit)

    def test_narrow_table_scrolls(self):
        for entry in self.result.results:
            entry.round_elapsed_ns = [ns * 10**7
                                      for ns in entry.round_elapsed_ns]
        self.widget.set_result(self.result)
        self.widget.resize(420, 360)
        self.widget.show()
        self.addCleanup(self.widget.close)
        self.app.processEvents()
        table = self.widget.table
        header = table.horizontalHeader()
        self.assertGreater(table.horizontalScrollBar().maximum(), 0)
        for column in range(1, table.columnCount()):
            self.assertGreaterEqual(header.sectionSize(column),
                                    header.sectionSizeHint(column))

    def test_chart_hover_and_resize(self):
        self.widget.set_result(self.result)
        chart = self.widget.chart
        for width in (420, 900):
            chart.setFixedSize(width, 200)
            self.hover_chart()
            self.assertIn('Median: 200 ns/call', chart.toolTip())
            self.assertIn('p95: 290 ns/call', chart.toolTip())
            self.assertIn('p5: 110 ns/call', chart.toolTip())
            self.assertIn('p25: 150 ns/call', chart.toolTip())
            self.assertIn('p75: 250 ns/call', chart.toolTip())
            self.assertIn('3 rounds, 4 calls/round; 2 warmups',
                          chart.toolTip())
        self.app.sendEvent(chart, QtCore.QEvent(QtCore.QEvent.Type.Leave))
        self.assertEqual(chart.toolTip(), '')

    def test_readable_ticks_cover_timings(self):
        cases = ((40.7013, [0, 10, 20, 30, 40, 50]),
                 (50, [0, 10, 20, 30, 40, 50]),
                 (580, [0, 200, 400, 600]),
                 (1.34568, [0, 0.5, 1, 1.5]),
                 (0.0407013, [0, 0.01, 0.02, 0.03, 0.04, 0.05]),
                 (0, [0, 0.2, 0.4, 0.6, 0.8, 1]))
        for maximum, expected in cases:
            with self.subTest(maximum=maximum):
                actual = self.widget.chart._ticks(maximum)
                self.assertEqual(len(actual), len(expected))
                for value, tick in zip(actual, expected):
                    self.assertAlmostEqual(value, tick)
                self.assertGreaterEqual(actual[-1], maximum)

    def test_replace_result_hides_previous_tooltip(self):
        self.addCleanup(QtWidgets.QToolTip.hideText)
        self.widget.set_result(self.result)
        self.hover_chart()
        self.assertTrue(QtWidgets.QToolTip.isVisible())
        self.assertIn('Median: 200 ns/call', QtWidgets.QToolTip.text())

        self.result.results[0].round_elapsed_ns = [800, 1600, 2400]
        with unittest.mock.patch.object(
                QtWidgets.QToolTip, 'hideText') as hide:
            self.widget.set_result(self.result)
        hide.assert_called_once()
        self.assertEqual(self.row_text(0)[1:3], ['400', '580'])
        self.assertEqual(self.widget.chart.toolTip(), '')

    def test_empty_output_and_zero_timings(self):
        empty_lhs = spec.OperandSpec((0, 2), (2, 1))
        self.result.spec = dataclasses.replace(self.result.spec, lhs=empty_lhs)
        for entry in self.result.results:
            if entry.status == 'measured':
                entry.max_abs_diff = None
                entry.relative_diff = None
                entry.round_elapsed_ns = [0, 0, 0]
        self.widget.set_result(self.result)
        self.assertEqual(self.row_text(0),
                         ['naive', '0', '0', '-', '-', 'measured'])
        self.hover_chart()
        self.assertIn('Median: 0 ns/call', self.widget.chart.toolTip())
        self.assertIn('p95: 0 ns/call', self.widget.chart.toolTip())

    def test_all_invalid_and_repeat(self):
        self.widget.set_result(self.result)
        self.hover_chart()
        self.assertIn('Median: 200 ns/call', self.widget.chart.toolTip())
        self.result.results = [
            results.KernelResult(entry.name, 'invalid',
                                 reason='Nonfinite NumPy output')
            for entry in self.result.results
        ]
        self.result.round_orders = [[], [], []]
        self.widget.set_result(self.result)
        self.assertEqual(self.row_text(0),
                         ['naive', '-', '-', '-', '-', 'invalid'])
        self.hover_chart()
        self.assertEqual(self.widget.chart.toolTip(), '')


if __name__ == '__main__':
    # Keep a child alive for tests that terminate a real process.
    time.sleep(60)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
