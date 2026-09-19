# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Exercise the inspector and results without mapping a top-level window."""

import pathlib
import tempfile
import time
import unittest
import unittest.mock

from solvcon import system
from solvcon.benchmark import artifact, matmul, spec

try:
    from PySide6 import QtCore, QtGui, QtTest, QtWidgets
except ImportError:
    QtWidgets = None
else:
    from solvcon.pilot import _benchmark_inspector


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class BenchmarkInspectorTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        self.widget = _benchmark_inspector.BenchmarkInspector()
        self.path = self.widget._path
        self.fields = self.widget.fields
        self.set_fields(lhs_shape='2, 3', lhs_strides='-3, 1',
                        rhs_shape='3, 2', rhs_strides='0, 1',
                        warmups='0', repetitions='1', rounds='1')
        for name, box in self.widget.kernels.items():
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

    def start_script(self, script):
        script = 'import sys\nsys.stdin.readline()\n' + script
        command = system.python_command('-c', script)
        with unittest.mock.patch.object(
                system, 'python_command', return_value=command):
            self.widget.run_button.click()

    def test_spec(self):
        self.set_fields(lhs_shape='2, 4, 3', lhs_strides='20, -5, 0',
                        rhs_shape='1, 3, 2', rhs_strides='0, 4, 1',
                        warmups='0', repetitions=str(2**32), rounds='7')
        self.widget.dtype.setCurrentText('complex64')
        self.widget.kernels['blas_gemm'].setChecked(True)
        self.assertEqual(self.widget.operation.count(), 1)
        self.assertEqual(self.widget.operation.currentText(), 'Matmul')
        self.assertEqual(self.widget.make_spec().to_dict(), {
            'operation': 'matmul', 'dtype': 'complex64',
            'lhs': {'shape': [2, 4, 3], 'strides': [20, -5, 0]},
            'rhs': {'shape': [1, 3, 2], 'strides': [0, 4, 1]},
            'sampling': {'warmups': 0, 'repetitions': 2**32, 'rounds': 7},
            'kernels': ['naive', 'blas_gemm']})

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
        boxes = self.widget.kernels
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
        self.widget.dtype.setCurrentText('complex64')
        self.assertTrue(winograd.isChecked())
        self.widget.run_button.click()
        self.wait_for_finish()
        self.assertTrue(winograd.isEnabled())
        self.assertFalse(boxes['blas_dot'].isEnabled())
        self.assertEqual(self.widget.control.status.text(), 'Completed')

    def test_kernel_rank_and_invalid_input(self):
        self.set_fields(lhs_shape='4', lhs_strides='-1',
                        rhs_shape='4', rhs_strides='0')
        boxes = self.widget.kernels
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
        self.widget.kernels['naive'].setChecked(False)
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
            self.assertEqual(artifact.load_artifact(self.path)['spec'],
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
        self.path.write_text('{}', encoding='utf8')
        self.widget.control.completed.emit(str(self.path))
        self.assertFalse(self.widget.save_button.isEnabled())
        self.assertIn('missing fields', self.widget.error.text())
        self.assertEqual(self.widget.results.summary.text(), snapshot)

    def test_recovery(self):
        control = self.widget.control
        for action in ('stop', 'close', 'failure'):
            with self.subTest(action=action):
                script = ('sys.exit(1)' if action == 'failure' else
                          'import time\ntime.sleep(60)')
                self.start_script(script)
                if action == 'stop':
                    control.stop_button.click()
                elif action == 'close':
                    self.widget.close()
                self.wait_for_finish()
                self.assertTrue(self.widget.inputs.isEnabled())
                self.assertTrue(self.widget.run_button.isEnabled())
        self.widget.run_button.click()
        self.wait_for_finish()
        self.assertEqual(control.status.text(), 'Completed')

    def test_save(self):
        self.widget.run_button.click()
        self.wait_for_finish()
        path = self.path.with_name('saved.json')
        with unittest.mock.patch.object(
                QtWidgets.QFileDialog, 'getSaveFileName') as dialog:
            dialog.return_value = ('', '')
            with unittest.mock.patch.object(
                    artifact, 'write_artifact') as write:
                self.widget.save_button.click()
                write.assert_not_called()
            dialog.return_value = (str(path), '')
            with unittest.mock.patch.object(
                    artifact, 'write_artifact', side_effect=OSError('Full')):
                self.widget.save_button.click()
            self.assertEqual(self.widget.error.text(), 'Full')
            self.fields['rounds'].setText('7')
            self.widget.save_button.click()
        self.assertEqual(self.widget.error.text(), '')
        self.assertEqual(artifact.load_artifact(path),
                         artifact.load_artifact(self.path))


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class TimingStatsTC(unittest.TestCase):
    def test_per_call_percentiles(self):
        cases = (([1200, 400, 800], 4, (200.0, 290.0)),
                 ([10, 30], 4, (5.0, 7.25)),
                 ([80], 5, (16.0, 16.0)),
                 ([0, 0], 4, (0.0, 0.0)),
                 ([], 4, (None, None)))
        for elapsed_ns, repetitions, expected in cases:
            with self.subTest(elapsed_ns=elapsed_ns):
                original = elapsed_ns.copy()
                timing = _benchmark_inspector.TimingStats.from_rounds(
                    elapsed_ns, repetitions)
                self.assertEqual((timing.median, timing.p95), expected)
                self.assertEqual(elapsed_ns, original)


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class BenchmarkResultsTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = pathlib.Path(self.directory.name) / 'result.json'
        self.widget = _benchmark_inspector.BenchmarkResults()
        self.addCleanup(self.widget.deleteLater)
        operand = spec.OperandSpec((2, 2), (2, 1))
        request = matmul.MatmulSpec(
            operand, operand, 'float64', spec.Sampling(2, 4, 3),
            ('naive', 'blas_dot', 'winograd'))
        self.result = {
            'spec': request.to_dict(),
            'round_orders': [['numpy', 'naive']] * 3,
            'results': [
                dict(name='naive', status='measured', reason=None,
                     max_abs_diff=0.25, relative_diff=0.125,
                     round_elapsed_ns=[400, 800, 1200]),
                dict(name='blas_dot', status='ineligible',
                     reason='Vectors only', max_abs_diff=None,
                     relative_diff=None,
                     round_elapsed_ns=[]),
                dict(name='winograd', status='invalid', reason='Nonfinite',
                     max_abs_diff=None, relative_diff=None,
                     round_elapsed_ns=[]),
                dict(name='numpy', status='measured', reason=None,
                     max_abs_diff=0.0, relative_diff=0.0,
                     round_elapsed_ns=[800, 1600, 2400])],
        }

    def load_result(self):
        artifact.write_artifact(self.result, self.path)
        self.widget.load(self.path)

    def row_text(self, row):
        table = self.widget.table
        return [table.item(row, col).text()
                for col in range(table.columnCount())]

    def chart_image(self, width=420):
        chart = self.widget.chart
        palette = chart.palette()
        for role, color in ((QtGui.QPalette.Base, 'white'),
                            (QtGui.QPalette.Text, 'black'),
                            (QtGui.QPalette.Highlight, 'red')):
            palette.setColor(role, QtGui.QColor(color))
        chart.setPalette(palette)
        chart.setFixedSize(width, 200)
        return chart.grab().toImage().scaled(chart.size())

    def hover_chart(self):
        point = QtCore.QPointF(120, 49)
        event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove, point, point,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.widget.chart, event)

    def test_summary_and_statuses(self):
        self.load_result()
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

    def test_empty_output_and_zero_timings(self):
        self.result['spec']['lhs']['shape'] = [0, 2]
        for row in self.result['results']:
            if row['status'] == 'measured':
                row.update(max_abs_diff=None, relative_diff=None,
                           round_elapsed_ns=[0, 0, 0])
        self.load_result()
        self.assertEqual(self.row_text(0),
                         ['naive', 'measured', '-', '-', '0', '0'])
        self.assertFalse(self.widget.chart.grab().isNull())

    def test_all_invalid_and_repeat(self):
        self.load_result()
        for row in self.result['results']:
            row.update(status='invalid', reason='Nonfinite NumPy output',
                       max_abs_diff=None, relative_diff=None,
                       round_elapsed_ns=[])
        self.result['round_orders'] = [[], [], []]
        self.load_result()
        self.assertEqual(self.row_text(0),
                         ['naive', 'invalid', '-', '-', '-', '-'])
        image = self.chart_image()
        self.assertEqual(image.pixelColor(120, 44), QtGui.QColor('white'))

    def test_reject_bad_artifact(self):
        self.load_result()
        self.path.write_text('{}', encoding='utf8')
        with self.assertRaises(artifact.ArtifactError):
            self.widget.load(self.path)
        self.assertEqual(self.row_text(0)[4:], ['200', '290'])

    def test_chart_hover_and_resize(self):
        self.load_result()
        chart = self.widget.chart
        for width in (420, 900):
            self.chart_image(width)
            self.hover_chart()
            self.assertIn('Median: 200 ns/call', chart.toolTip())
            self.assertIn('p95: 290 ns/call', chart.toolTip())
            self.assertIn('3 rounds, 4 calls/round; 2 warmups',
                          chart.toolTip())
        self.app.sendEvent(chart, QtCore.QEvent(QtCore.QEvent.Type.Leave))
        self.assertEqual(chart.toolTip(), '')

    def test_chart_bar_and_whisker_positions(self):
        self.load_result()
        # Logical pixels for medians of 200/400 ns and p95s of 290/580 ns.
        cases = ((420, 44, 175, 205), (420, 126, 241, 300),
                 (900, 44, 341, 445), (900, 126, 572, 780))
        for width, pixel_y, median_x, p95_x in cases:
            with self.subTest(width=width, pixel_y=pixel_y):
                image = self.chart_image(width)
                for pixel_x, color in ((111, 'red'),
                                       (median_x - 3, 'red'),
                                       (median_x + 3, 'white'),
                                       (p95_x - 3, 'white'),
                                       (p95_x, 'black'),
                                       (p95_x + 3, 'white')):
                    self.assertEqual(image.pixelColor(pixel_x, pixel_y),
                                     QtGui.QColor(color),
                                     f'({pixel_x}, {pixel_y})')

    def test_reload_hides_previous_tooltip(self):
        self.addCleanup(QtWidgets.QToolTip.hideText)
        self.load_result()
        self.chart_image()
        self.hover_chart()
        self.assertTrue(QtWidgets.QToolTip.isVisible())
        self.assertIn('Median: 200 ns/call', QtWidgets.QToolTip.text())

        self.result['results'][0]['round_elapsed_ns'] = [800, 1600, 2400]
        self.load_result()
        self.assertEqual(self.row_text(0)[4:], ['400', '580'])
        QtTest.QTest.qWait(350)
        self.assertFalse(QtWidgets.QToolTip.isVisible())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
