# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Exercise benchmark widgets without mapping a top-level window."""

import dataclasses
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
import unittest.mock

from solvcon import system
from solvcon.benchmark import matmul, results, spec

try:
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
except ImportError:
    QtWidgets = None
else:
    from solvcon.pilot.benchmark import _inspector
    from solvcon.pilot.benchmark import _run


@dataclasses.dataclass
class WorkerStub:
    """Emit configured data in a real process for controller tests."""

    events: tuple = ()
    output: str = ''
    error: str = ''
    exit_code: int | None = None
    environment: dict = dataclasses.field(default_factory=dict)

    def command(self):
        return system.python_command(
            __file__, json.dumps(dataclasses.asdict(self)))

    def run(self):
        sys.stdin.readline()
        actual = {name: os.environ.get(name) for name in self.environment}
        if actual != self.environment:
            raise RuntimeError(f'Unexpected worker environment: {actual}')
        for event in self.events:
            print(json.dumps(event), flush=True)
        sys.stdout.write(self.output)
        sys.stdout.flush()
        sys.stderr.write(self.error)
        sys.stderr.flush()
        if self.exit_code is not None:
            return self.exit_code
        # Stay alive until Stop or error handling terminates the process.
        time.sleep(60)
        return 0


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
        self.control.stop()
        self.wait_for(lambda: not self.control.running)
        self.control.close()
        self.control.deleteLater()
        self.directory.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate(), self.control.status.text())

    def start_worker(self, worker, **options):
        command = worker.command()
        with unittest.mock.patch.object(
            system, 'python_command', return_value=command,
        ):
            self.control.start(self.spec, self.path, **options)

    def start_events(self, *events):
        self.start_worker(WorkerStub(events=events))

    def assert_finished(self, kind):
        self.wait_for(lambda: not self.control.running)
        self.assertEqual([event[0] for event in self.events], [kind])
        self.assertEqual(
            self.control._process.state(),
            QtCore.QProcess.ProcessState.NotRunning,
        )
        self.assertFalse(self.control.stop_button.isEnabled())
        self.assertFalse(self.control._timer.isActive())
        success = kind == 'result'
        self.assertEqual(self.control.progress.value(), int(success))
        self.assertEqual(self.control.progress.isTextVisible(), success)

    def assert_failed(self, message):
        self.assert_finished('error')
        self.assertIn(message, self.events[0][1])

    def assert_reusable(self):
        self.events.clear()
        self.control.start(self.spec, self.path)
        self.assert_finished('result')
        document = results.load_artifact(self.path).to_dict()
        self.assertEqual(document['spec'], self.spec.to_dict())

    def test_collect_and_repeat(self):
        for _ in range(2):
            self.events.clear()
            self.control.start(self.spec, self.path)
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                self.control.start(self.spec, self.path)
            self.assert_finished('result')
            self.assertEqual(self.events[0][1], str(self.path))
            self.assertEqual(results.load_artifact(self.path).spec.to_dict(),
                             self.spec.to_dict())

    def test_progress_stop_and_recover(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        self.start_events(event)
        self.wait_for(lambda: self.control.status.text() == 'Timing: naive')
        self.assertEqual(self.events, [])
        self.assertEqual(self.control.progress.value(), 16)
        self.assertEqual(self.control.progress.text(), '16% (1/6 units)')
        self.control.stop_button.click()
        self.assert_finished('stopped')
        self.assert_reusable()

    def test_waits_for_complete_event(self):
        self.start_worker(WorkerStub())
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
        process = self.control._process
        with (
            unittest.mock.patch.object(
                process, 'canReadLine', side_effect=stream.canReadLine),
            unittest.mock.patch.object(
                process, 'readLine', side_effect=stream.readLine),
        ):
            for chunk, expected in chunks:
                unread = stream.pos()
                stream.seek(stream.size())
                stream.write(chunk)
                stream.seek(unread)
                self.control._read_stdout()
                self.assertEqual(self.control.status.text(), expected)
                self.assertEqual(self.control._error, '')
                self.assertEqual(self.events, [])
        self.assertTrue(stream.atEnd())
        self.assertEqual(self.control.progress.text(), '16% (1/6 units)')

    def test_elapsed_while_running(self):
        self.start_worker(WorkerStub())
        self.wait_for(
            lambda: self.control.elapsed.text() != 'Elapsed: 0.0 s')
        self.assertTrue(self.control.running)
        self.assertEqual(self.events, [])

    def test_large_progress_counts(self):
        self.start_worker(WorkerStub())
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
        self.start_events({
            'type': 'progress', 'phase': 'timing', 'kernel': 'numpy',
            'completed': 6, 'total': 6,
        })
        self.wait_for(lambda: self.control.progress.isTextVisible())
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
        cases = (
            ('missing_count', {'completed': None}),
            ('boolean_count', {'completed': True}),
            ('negative_count', {'completed': -1}),
            ('exceeds_total', {'completed': 7}),
            ('zero_total', {'total': 0}),
            ('noninteger_total', {'total': 6.0}),
            ('unknown_kernel', {'kernel': 'missing'}),
            ('finishing_with_counts', {'phase': 'finishing', 'kernel': None}),
        )
        for name, change in cases:
            with self.subTest(case=name):
                self.events.clear()
                self.start_events({**event, **change})
                self.assert_failed('invalid worker progress')
                self.assert_reusable()

    def test_rejects_inconsistent_progress(self):
        event = {
            'type': 'progress', 'phase': 'timing', 'kernel': 'naive',
            'completed': 1, 'total': 6,
        }
        values = []
        self.control.progress.valueChanged.connect(values.append)
        cases = (
            ('regressing_count', {'completed': 0}),
            ('changing_total', {'total': 7}),
        )
        for name, change in cases:
            with self.subTest(case=name):
                self.events.clear()
                values.clear()
                self.start_events(event, {**event, **change})
                self.assert_failed('invalid worker progress counts')
                self.assertIn(16, values)
                self.assert_reusable()

    def test_thread_isolation(self):
        names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
        inherited = dict.fromkeys(names, '2')
        event = {'type': 'result', 'artifact_path': 'ok'}
        for threads, expected in ((3, '3'), (None, '2')):
            with self.subTest(threads=threads):
                self.events.clear()
                worker = WorkerStub(
                    events=(event,), exit_code=0,
                    environment=dict.fromkeys(names, expected))
                with unittest.mock.patch.dict(os.environ, inherited):
                    self.start_worker(worker, threads=threads)
                    self.assert_finished('result')
                    parent = {name: os.environ[name] for name in names}
                    self.assertEqual(parent, inherited)

    def test_invalid_threads(self):
        for threads in (0, -1, True, 1.5, '2'):
            with self.subTest(threads=threads):
                with self.assertRaises(ValueError) as caught:
                    self.control.start(self.spec, self.path, threads=threads)
                self.assertEqual(str(caught.exception),
                                 'threads must be a positive integer')
                self.assertFalse(self.control.running)

    def assert_error_recovery(self, cases):
        for worker, message in cases:
            with self.subTest(message=message):
                self.events.clear()
                self.start_worker(worker)
                self.assert_failed(message)
                self.assert_reusable()

    def test_protocol_error_recovery(self):
        error = {'type': 'error', 'message': 'bad spec'}
        self.assert_error_recovery((
            (WorkerStub(output='not json\n'), 'protocol'),
            (WorkerStub(events=({},)), 'unknown'),
            (WorkerStub(events=(error,)), 'bad spec'),
        ))

    def test_exit_error_recovery(self):
        result = {'type': 'result', 'artifact_path': 'fake'}
        self.assert_error_recovery((
            (WorkerStub(output='{}', exit_code=0), 'incomplete'),
            (WorkerStub(exit_code=0), 'without a result'),
            (WorkerStub(error='native crash', exit_code=3),
             'native crash'),
            (WorkerStub(events=(result,), exit_code=3), 'code 3'),
        ))

    def test_crash(self):
        self.start_worker(WorkerStub())
        self.wait_for(lambda: self.control._process.state() ==
                      QtCore.QProcess.ProcessState.Running)
        self.control._process.kill()
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

    def test_stop_during_start_and_close(self):
        for action in (self.control.stop, self.control.close):
            self.events.clear()
            self.start_worker(WorkerStub())
            action()
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
        self.widget.control.stop()
        self.wait_for_finish()
        self.widget.close()
        self.widget.deleteLater()

    def set_fields(self, **values):
        for name, value in values.items():
            self.fields[name].setText(value)

    def wait_for_finish(self):
        deadline = time.monotonic() + 15
        while self.widget.control.running and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertFalse(self.widget.control.running)

    def start_worker(self, worker):
        command = worker.command()
        with unittest.mock.patch.object(
                system, 'python_command', return_value=command):
            self.widget.run_button.click()

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
        expected = (
            'Each timed kernel runs 3 untimed warmups, then 2 rounds of '
            '7 calls each. NumPy uses the same schedule.')
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
        self.widget.run_button.click()
        self.wait_for_finish()
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
            self.widget.run_button.click()
            self.assertTrue(self.widget.control.running)
            self.assertFalse(self.widget.inputs.isEnabled())
            self.assertFalse(self.widget.run_button.isEnabled())
            self.assertFalse(self.widget.save_button.isEnabled())
            self.wait_for_finish()
            self.assertEqual(self.widget.control.status.text(), 'Completed')
            self.assertEqual(results.load_artifact(self.path).spec.to_dict(),
                             expected)
            self.assertTrue(self.widget.inputs.isEnabled())
            self.assertTrue(self.widget.run_button.isEnabled())
            self.assertTrue(self.widget.save_button.isEnabled())
            table = self.widget.results.table
            self.assertEqual(table.rowCount(), 2)
            self.assertEqual(table.item(0, 0).text(), 'naive')
            self.assertEqual(table.item(1, 0).text(), 'numpy')

    def test_result_uses_artifact(self):
        self.widget.run_button.click()
        self.wait_for_finish()
        snapshot = self.widget.results.summary.text()
        self.set_fields(lhs_shape='4, 3', lhs_strides='3, 1')
        self.widget.control.completed.emit(str(self.path))
        self.assertEqual(self.widget.results.summary.text(), snapshot)
        self.assertIn('A (2, 3) strides (-3, 1)', snapshot)

    def test_invalid_result_disables_save(self):
        self.widget.run_button.click()
        self.wait_for_finish()
        self.assertTrue(self.widget.save_button.isEnabled())
        snapshot = self.widget.results.summary.text()
        table = self.widget.results.table
        median = table.item(0, 4).text()
        self.path.write_text('{}', encoding='utf8')
        self.widget.control.completed.emit(str(self.path))
        self.assertFalse(self.widget.save_button.isEnabled())
        self.assertIn('missing fields', self.widget.error.text())
        self.assertEqual(self.widget.results.summary.text(), snapshot)
        self.assertEqual(table.item(0, 4).text(), median)

    def test_stop_and_close_recovery(self):
        control = self.widget.control
        actions = (control.stop_button.click, self.widget.close)
        for action in actions:
            with self.subTest(action=action.__name__):
                self.start_worker(WorkerStub())
                action()
                self.wait_for_finish()
                self.assertTrue(self.widget.inputs.isEnabled())
                self.assertTrue(self.widget.run_button.isEnabled())
                self.widget.run_button.click()
                self.wait_for_finish()
                self.assertEqual(control.status.text(), 'Completed')

    def test_failure_recovery(self):
        self.start_worker(WorkerStub(exit_code=1))
        self.wait_for_finish()
        self.assertTrue(self.widget.inputs.isEnabled())
        self.assertTrue(self.widget.run_button.isEnabled())
        self.widget.run_button.click()
        self.wait_for_finish()
        self.assertEqual(self.widget.control.status.text(), 'Completed')

    def test_save(self):
        self.widget.run_button.click()
        self.wait_for_finish()
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
        self.widget = _inspector.ResultView(_inspector.MatmulForm.describe)
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
                         ['naive', 'measured', '0.25', '0.125', '200', '290'])
        self.assertEqual(self.row_text(1),
                         ['blas_dot', 'ineligible', '-', '-', '-', '-'])
        self.assertEqual(self.row_text(2),
                         ['winograd', 'invalid', '-', '-', '-', '-'])
        self.assertEqual(self.row_text(3),
                         ['numpy', 'measured', '0', '0', '400', '580'])
        self.assertIn('Vectors only', self.widget.table.item(1, 1).toolTip())
        self.assertIn('Nonfinite', self.widget.table.item(2, 1).toolTip())

    def test_chart_hover_and_resize(self):
        self.widget.set_result(self.result)
        chart = self.widget.chart
        for width in (420, 900):
            chart.setFixedSize(width, 200)
            self.hover_chart()
            self.assertIn('Median: 200 ns/call', chart.toolTip())
            self.assertIn('p95: 290 ns/call', chart.toolTip())
            self.assertIn('3 rounds, 4 calls/round; 2 warmups',
                          chart.toolTip())
        self.app.sendEvent(chart, QtCore.QEvent(QtCore.QEvent.Type.Leave))
        self.assertEqual(chart.toolTip(), '')

    def test_replace_result_hides_previous_tooltip(self):
        self.addCleanup(QtWidgets.QToolTip.hideText)
        self.widget.set_result(self.result)
        self.hover_chart()
        self.assertTrue(QtWidgets.QToolTip.isVisible())
        self.assertIn('Median: 200 ns/call', QtWidgets.QToolTip.text())

        self.result.results[0].round_elapsed_ns = [800, 1600, 2400]
        self.widget.set_result(self.result)
        self.assertEqual(self.row_text(0)[4:], ['400', '580'])
        QtTest.QTest.qWait(350)
        self.assertFalse(QtWidgets.QToolTip.isVisible())

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
                         ['naive', 'measured', '-', '-', '0', '0'])
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
                         ['naive', 'invalid', '-', '-', '-', '-'])
        self.hover_chart()
        self.assertEqual(self.widget.chart.toolTip(), '')


if __name__ == '__main__':
    sys.exit(WorkerStub(**json.loads(sys.argv[1])).run())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
