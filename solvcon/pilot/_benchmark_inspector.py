# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Edit and run one exact matmul comparison."""

import functools
import pathlib
import tempfile

from PySide6 import QtCore, QtWidgets

from solvcon.benchmark import artifact, matmul, spec
from . import _benchmark


def _integers(text, name):
    try:
        return tuple(int(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise spec.SpecError(
            f'{name}: enter comma-separated integers') from exc


class BenchmarkInspector(QtWidgets.QMdiSubWindow):
    """Run one comparison in Pilot and export its last completed result."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Benchmark Inspector')
        self._directory = tempfile.TemporaryDirectory()
        self._path = pathlib.Path(self._directory.name) / 'result.json'
        self.destroyed.connect(self._directory.cleanup)
        self._closing = False
        self._build_inputs()
        self._build_controls()
        self.dtype.currentTextChanged.connect(self._update_kernels)
        for name in ('lhs_shape', 'lhs_strides', 'rhs_shape', 'rhs_strides'):
            self.fields[name].textChanged.connect(self._update_kernels)
        self._update_kernels()
        self.resize(1000, 520)

    def make_spec(self):
        inputs = self._matmul_inputs()
        kernels = tuple(name for name, box in self.kernels.items()
                        if box.isChecked())
        return matmul.MatmulSpec(
            inputs.lhs, inputs.rhs, inputs.dtype, self._sampling(), kernels)

    def _matmul_inputs(self):
        operands = []
        for name in ('lhs', 'rhs'):
            shape = _integers(self.fields[f'{name}_shape'].text(),
                              f'{name} shape')
            strides = _integers(self.fields[f'{name}_strides'].text(),
                                f'{name} strides')
            operands.append(spec.OperandSpec(shape, strides))
        return matmul.MatmulInputs(
            lhs=operands[0], rhs=operands[1],
            dtype=self.dtype.currentText())

    def _build_inputs(self):
        self.inputs = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(self.inputs)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.dtype = QtWidgets.QComboBox(self.inputs)
        self.dtype.addItems(matmul.MATMUL_DTYPES)
        self.dtype.setCurrentText('float64')
        self.dtype.setToolTip('Element type for both inputs and the result.')
        self.fields = {
            name: QtWidgets.QLineEdit(value, self.inputs)
            for name, value in (
                ('lhs_shape', '64, 64'), ('lhs_strides', '64, 1'),
                ('rhs_shape', '64, 64'), ('rhs_strides', '64, 1'),
                ('threads', '1'), ('warmups', '2'),
                ('repetitions', '5'), ('rounds', '5'))
        }
        form.addRow('Data type', self.dtype)
        for name, label in (('lhs', 'A'), ('rhs', 'B')):
            form.addRow(f'{label} shape', self.fields[f'{name}_shape'])
            form.addRow(f'{label} element strides',
                        self.fields[f'{name}_strides'])
        form.addRow('Worker threads', self.fields['threads'])

        for name in ('lhs_shape', 'rhs_shape'):
            self.fields[name].setToolTip(
                'Full shape, including batch axes: e.g. 2, 32, 64.')
        for name in ('lhs_strides', 'rhs_strides'):
            self.fields[name].setToolTip(
                'One stride per axis, in elements. Negative and zero '
                'strides are allowed. Values are used exactly as entered.')
        self.fields['threads'].setToolTip(
            'Requested BLAS/OpenMP threads in the worker. Libraries may '
            'use fewer threads. Pilot keeps its current thread settings.')

        sampling = QtWidgets.QGridLayout()
        for col, (name, label, tooltip) in enumerate((
                ('warmups', 'Warmups',
                 'Untimed calls per kernel before timing; zero is allowed.'),
                ('repetitions', 'Calls per round',
                 'Consecutive calls timed together per round. '
                 'Divide the block time by this count for time per call.'),
                ('rounds', 'Rounds', 'Measurement rounds per kernel. '
                 'Each round records one sample; '
                 'kernel order is balanced across rounds.'))):
            sampling.addWidget(QtWidgets.QLabel(label), 0, col)
            sampling.addWidget(self.fields[name], 1, col)
            sampling.setColumnStretch(col, 1)
            self.fields[name].setToolTip(tooltip)
            self.fields[name].textChanged.connect(self._update_help)
        self.sampling_help = QtWidgets.QLabel(self.inputs)
        self.sampling_help.setWordWrap(True)
        sampling.addWidget(self.sampling_help, 2, 0, 1, 3)
        form.addRow('Sampling', sampling)
        self._update_help()
        self._build_kernels(form)

    def _build_kernels(self, form):
        layout = QtWidgets.QGridLayout()
        self.kernels = {}
        self._kernel_choices = {}
        self._kernel_tooltips = {}
        choices = (
            ('naive', 'Naive', 'Direct native loop for any valid input.'),
            ('blas_dot', 'BLAS DOT', 'Vector @ vector.'),
            ('blas_gevm', 'BLAS GEVM', 'Vector @ matrix.'),
            ('blas_gemv', 'BLAS GEMV', 'Matrix @ vector.'),
            ('blas_gemm', 'BLAS GEMM', 'Matrix @ matrix, including batches.'),
            ('winograd', 'Winograd', 'Unbatched matrices with positive, '
             'even M, K and N.'))
        for index, (name, label, tooltip) in enumerate(choices):
            box = QtWidgets.QCheckBox(label, self.inputs)
            if name != 'naive':
                tooltip += ' Requires a supported dtype and BLAS backend.'
            box.setToolTip(tooltip)
            box.setChecked(True)
            self._kernel_choices[name] = True
            self._kernel_tooltips[name] = tooltip
            box.toggled.connect(
                functools.partial(self._remember_kernel, name))
            layout.addWidget(box, index // 3, index % 3)
            self.kernels[name] = box
        form.addRow('Kernels', layout)
        kernel_help = QtWidgets.QLabel(
            'NumPy is always included for timing and numerical comparison. '
            'Unavailable kernels are disabled; hover for the reason.',
            self.inputs)
        kernel_help.setWordWrap(True)
        layout.addWidget(kernel_help, 2, 0, 1, 3)

    def _remember_kernel(self, name, checked):
        self._kernel_choices[name] = checked

    def _update_kernels(self):
        try:
            reasons = self._matmul_inputs().kernel_eligibility()
        except ValueError as exc:
            reasons = dict.fromkeys(self.kernels, str(exc))
        for name, box in self.kernels.items():
            reason = reasons[name]
            with QtCore.QSignalBlocker(box):
                box.setEnabled(reason is None)
                box.setChecked(reason is None and self._kernel_choices[name])
            tooltip = self._kernel_tooltips[name]
            if reason is not None:
                tooltip += '\nUnavailable: ' + reason
            box.setToolTip(tooltip)

    def _build_controls(self):
        self.operation = QtWidgets.QComboBox(self)
        self.operation.addItem('Matmul', 'matmul')
        self.operation.setToolTip('Currently supports Matmul.')
        self.error = QtWidgets.QLabel(self)
        self.error.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.error.setWordWrap(True)
        self.error.hide()
        self.run_button = QtWidgets.QPushButton('Run', self)
        self.control = _benchmark.BenchmarkControl(self)
        self.control.status.setWordWrap(True)
        self.control.layout().setContentsMargins(0, 0, 0, 0)
        self.save_button = QtWidgets.QPushButton('Save result...', self)
        self.save_button.setEnabled(False)

        self.run_button.clicked.connect(self._run)
        self.save_button.clicked.connect(self._save)
        for signal in (self.control.completed, self.control.failed,
                       self.control.stopped):
            signal.connect(self._finished)
        self.control.completed.connect(
            lambda: self.save_button.setEnabled(True))

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel('Operation'))
        header.addWidget(self.operation)
        header.addStretch(1)
        header.addWidget(self.control.stop_button)

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.save_button)
        actions.addStretch(1)

        content = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(content)
        layout.addLayout(header)
        layout.addWidget(self.inputs)
        layout.addWidget(self.error)
        layout.addLayout(actions)
        layout.addWidget(self.control)
        layout.addStretch(1)
        self.setWidget(content)

    def _save(self):
        self._show_error('')
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save benchmark result', 'matmul-result.json',
            'JSON files (*.json)')
        if not path:
            return
        try:
            artifact.write_artifact(artifact.load_artifact(self._path), path)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))

    def _show_error(self, text):
        self.error.setText(text)
        self.error.setVisible(bool(text))

    def _update_help(self):
        try:
            sampling = self._sampling()
        except ValueError as exc:
            text = str(exc)
        else:
            text = (
                f'Each timed kernel runs {sampling.warmups} untimed warmups, '
                f'then {sampling.rounds} rounds of '
                f'{sampling.repetitions} calls each. '
                'NumPy uses the same schedule.')
        self.sampling_help.setText(text)

    def _sampling(self):
        return spec.Sampling(**{
            name: self._count(name)
            for name in ('warmups', 'repetitions', 'rounds')})

    def _count(self, name):
        values = _integers(self.fields[name].text(), name)
        if len(values) != 1:
            raise spec.SpecError(f'{name}: enter one integer')
        return values[0]

    def _run(self):
        if self.control.running:
            return
        self._show_error('')
        try:
            request = self.make_spec()
            threads = self._count('threads')
            self.control.start(request, self._path, threads=threads)
        except (ValueError, RuntimeError) as exc:
            self._show_error(str(exc))
            return
        self.save_button.setEnabled(False)
        self.operation.setEnabled(False)
        self.inputs.setEnabled(False)
        self.run_button.setEnabled(False)

    def _finished(self):
        self.operation.setEnabled(True)
        self.inputs.setEnabled(True)
        self.run_button.setEnabled(True)
        if self._closing:
            self.close()

    def closeEvent(self, event):
        if self.control.running:
            self._closing = True
            self.control.stop()
            event.ignore()
        else:
            self._closing = False
            super().closeEvent(event)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
