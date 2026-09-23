# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Edit, run, and display one exact matmul comparison."""

import dataclasses
import functools
import pathlib
import tempfile

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from solvcon.benchmark import artifact, matmul, spec
from . import _run


def _integers(text, name):
    try:
        return tuple(int(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise spec.SpecError(
            f'{name}: enter comma-separated integers') from exc


def _format_number(value):
    return '-' if value is None else f'{value:.6g}'


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
        self.resize(1000, 850)

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

    def _sampling(self):
        return spec.Sampling(**{
            name: self._count(name)
            for name in ('warmups', 'repetitions', 'rounds')})

    def _count(self, name):
        values = _integers(self.fields[name].text(), name)
        if len(values) != 1:
            raise spec.SpecError(f'{name}: enter one integer')
        return values[0]

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
            box.toggled.connect(functools.partial(self._remember_kernel, name))
            layout.addWidget(box, index // 3, index % 3)
            self.kernels[name] = box
        form.addRow('Kernels', layout)
        kernel_help = QtWidgets.QLabel(
            'NumPy is always included for timing and numerical comparison. '
            'Unavailable kernels are disabled; hover for the reason.',
            self.inputs)
        kernel_help.setWordWrap(True)
        layout.addWidget(kernel_help, 2, 0, 1, 3)

    def _build_controls(self):
        self.operation = QtWidgets.QComboBox(self)
        self.operation.addItem('Matmul', 'matmul')
        self.operation.setToolTip('Currently supports Matmul.')
        self.error = QtWidgets.QLabel(self)
        self.error.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.error.setWordWrap(True)
        self.error.hide()
        self.run_button = QtWidgets.QPushButton('Run', self)
        self.control = _run.RunPanel(self)
        self.control.status.setWordWrap(True)
        self.control.layout().setContentsMargins(0, 0, 0, 0)
        self.save_button = QtWidgets.QPushButton('Save result...', self)
        self.save_button.setEnabled(False)

        self.run_button.clicked.connect(self._run)
        self.save_button.clicked.connect(self._save)
        for signal in (self.control.completed, self.control.failed,
                       self.control.stopped):
            signal.connect(self._finished)
        self.control.completed.connect(self._completed)

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
        self.results = ResultView(self)
        layout.addWidget(self.results, 1)
        self.setWidget(content)

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

    def _remember_kernel(self, name, checked):
        self._kernel_choices[name] = checked

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

    def _completed(self, path):
        try:
            self.results.load(path)
        except (OSError, ValueError) as exc:
            self.save_button.setEnabled(False)
            self._show_error(str(exc))
            return
        self.save_button.setEnabled(True)

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

    def closeEvent(self, event):
        if self.control.running:
            self._closing = True
            self.control.stop()
            event.ignore()
        else:
            self._closing = False
            super().closeEvent(event)


class ResultView(QtWidgets.QWidget):
    """Keep displayed results independent of the current input controls."""

    HEADERS = ('Kernel', 'Status', 'Max abs diff', 'Relative diff',
               'Median (ns/call)', 'p95 (ns/call)')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = QtWidgets.QLabel('No completed result', self)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.chart = TimingChart(self)
        self.table = QtWidgets.QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(100)
        self.table.horizontalHeaderItem(2).setToolTip(
            'Maximum absolute difference from the NumPy result.')
        self.table.horizontalHeaderItem(3).setToolTip(
            'Max abs diff / max(abs(NumPy result)); "-" means unavailable.')
        legend = QtWidgets.QLabel(
            'Per-call round averages: bar = median, whisker = p95. '
            'Lower is faster; hover for details.', self)
        legend.setWordWrap(True)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.table)
        splitter.setChildrenCollapsible(False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary)
        layout.addWidget(legend)
        layout.addWidget(splitter, 1)

    def load(self, path):
        result = artifact.load_artifact(path)
        rows = self._make_rows(result)
        specification = result['spec']
        lhs, rhs = specification['lhs'], specification['rhs']
        self.summary.setText(
            f'Last completed result: {specification["dtype"]}; '
            f'A {tuple(lhs["shape"])} strides {tuple(lhs["strides"])}; '
            f'B {tuple(rhs["shape"])} strides {tuple(rhs["strides"])}')
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            timing = row['timing']
            values = (row['name'], row['status'],
                      _format_number(row['max_abs_diff']),
                      _format_number(row['relative_diff']),
                      _format_number(timing.median),
                      _format_number(timing.p95))
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(row['tooltip'])
                self.table.setItem(index, column, item)
        self.table.resizeColumnsToContents()
        self.chart.set_rows([row for row in rows
                             if row['status'] == 'measured'])

    @staticmethod
    def _make_rows(result):
        sampling = result['spec']['sampling']
        repetitions = sampling['repetitions']
        rows = []
        for entry in result['results']:
            timing = TimingStats.from_rounds(
                entry['round_elapsed_ns'], repetitions)
            row = dict(entry, timing=timing)
            row['tooltip'] = (
                f'{entry["name"]}: {entry["status"]}\n'
                f'Median: {_format_number(timing.median)} ns/call\n'
                f'p95: {_format_number(timing.p95)} ns/call\n'
                f'Max abs diff: {_format_number(entry["max_abs_diff"])}\n'
                f'Relative diff: {_format_number(entry["relative_diff"])}\n'
                f'{sampling["rounds"]} rounds, {repetitions} calls/round; '
                f'{sampling["warmups"]} warmups\n'
                'Percentiles use per-call round averages.'
            )
            if entry['reason']:
                row['tooltip'] += f'\n{entry["reason"]}'
            rows.append(row)
        return rows


class TimingChart(QtWidgets.QWidget):
    """Show median bars and p95 whiskers for per-call round averages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._regions = []
        self._empty_text = 'Run a benchmark to compare kernel timings'
        self.setMouseTracking(True)
        self.setMinimumSize(420, 180)

    def set_rows(self, rows):
        self._rows = rows
        self._regions = []
        self._empty_text = 'No measured timings in the completed result'
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()
        self.update()

    def paintEvent(self, event):
        self._regions = []
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        align = QtCore.Qt.AlignmentFlag
        if not self._rows:
            painter.drawText(self.rect(), align.AlignCenter, self._empty_text)
            return

        maximum = max(row['timing'].p95 for row in self._rows) or 1
        label_width, value_width = 100, 110
        padding, gap, top, axis_height = 4, 6, 8, 24
        left = padding + label_width + gap
        right = gap + value_width + padding
        plot = QtCore.QRectF(
            left, top, self.width() - left - right,
            self.height() - top - axis_height - padding,
        )
        axis = QtCore.QRectF(
            plot.left(), plot.bottom(), plot.width(), axis_height,
        )
        height = plot.height() / len(self._rows)
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawText(axis, align.AlignLeft | align.AlignVCenter, '0')
        painter.drawText(
            axis, align.AlignRight | align.AlignVCenter,
            f'{_format_number(maximum)} ns/call',
        )
        for index, row in enumerate(self._rows):
            timing = row['timing']
            rect = QtCore.QRectF(0, plot.top() + index * height,
                                 self.width(), height)
            self._regions.append((rect, row['tooltip']))
            painter.drawText(
                QtCore.QRectF(padding, rect.top(), label_width, height),
                align.AlignRight | align.AlignVCenter, row['name'],
            )
            middle = rect.center().y()
            bar_height = min(20, height * 0.6)
            median_x = plot.left() + plot.width() * timing.median / maximum
            p95_x = plot.left() + plot.width() * timing.p95 / maximum
            painter.fillRect(
                QtCore.QRectF(plot.left(), middle - bar_height / 2,
                              median_x - plot.left(), bar_height),
                self.palette().highlight())
            painter.drawLine(QtCore.QPointF(median_x, middle),
                             QtCore.QPointF(p95_x, middle))
            painter.drawLine(QtCore.QPointF(p95_x, middle - bar_height / 3),
                             QtCore.QPointF(p95_x, middle + bar_height / 3))
            painter.drawText(
                QtCore.QRectF(
                    plot.right() + gap, rect.top(), value_width, height,
                ),
                align.AlignLeft | align.AlignVCenter,
                _format_number(timing.median),
            )

    def mouseMoveEvent(self, event):
        tooltip = next((text for rect, text in self._regions
                        if rect.contains(event.position())), '')
        self.setToolTip(tooltip)
        if tooltip:
            QtWidgets.QToolTip.showText(
                event.globalPosition().toPoint(), tooltip, self)
        else:
            QtWidgets.QToolTip.hideText()

    def leaveEvent(self, event):
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)


@dataclasses.dataclass(frozen=True)
class TimingStats:
    """Summarize per-call round averages in ns/call.

    :ivar median: Median, or ``None`` when no samples are available.
    :ivar p95: 95th percentile, or ``None`` when no samples are available.
    """

    median: float | None = None
    p95: float | None = None

    @classmethod
    def from_rounds(cls, elapsed_ns, repetitions):
        """Summarize validated rounds; no samples yield unavailable timings."""
        if not elapsed_ns:
            return cls()
        samples = np.array(elapsed_ns, dtype='float64')
        samples /= repetitions
        median, p95 = np.percentile(samples, [50, 95], method='linear')
        return cls(float(median), float(p95))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
