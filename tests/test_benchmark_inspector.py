# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Exercise the inspector without mapping a top-level window."""

import time
import unittest
import unittest.mock

from solvcon import system
from solvcon.benchmark import artifact

try:
    from PySide6 import QtTest, QtWidgets
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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
