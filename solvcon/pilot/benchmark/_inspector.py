# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Edit, run, and display one exact kernel comparison."""

import functools
import math
import pathlib
import tempfile

from PySide6 import QtCore, QtGui, QtWidgets

from solvcon.benchmark import matmul, results, spec
from . import _run


def _integers(text, name):
    try:
        return tuple(int(item.strip()) for item in text.split(','))
    except ValueError as exc:
        raise spec.SpecError(
            f'{name}: enter comma-separated integers') from exc


def _format_number(value, scale=1):
    return '-' if value is None else f'{value / scale:.6g}'


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
        self.resize(1000, 850)

    def make_spec(self):
        """Read validated inputs and sampling without allocating operands."""
        return self.form.make_spec(self._sampling())

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
        self.inputs = QtWidgets.QGroupBox('Inputs and sampling', self)
        layout = QtWidgets.QFormLayout(self.inputs)
        layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form = MatmulForm(layout)
        self.fields = {
            name: QtWidgets.QLineEdit(value, self.inputs)
            for name, value in (
                ('threads', '1'), ('warmups', '2'),
                ('repetitions', '5'), ('rounds', '5'))
        }
        layout.addRow('Worker threads', self.fields['threads'])
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
        self.sampling_help.setToolTip(
            'Warmups are untimed. Each round times consecutive calls, then '
            'divides by the call count. NumPy uses the same schedule.')
        sampling.addWidget(self.sampling_help, 2, 0, 1, 3)
        layout.addRow('Sampling', sampling)
        self._update_help()
        self.form.add_kernels(layout)

    def _build_controls(self):
        self.operation = QtWidgets.QComboBox(self)
        self.operation.addItem(self.form.LABEL, self.form.OPERATION)
        self.operation.setToolTip(f'Currently supports {self.form.LABEL}.')
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

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.control.stop_button)
        actions.addStretch(1)
        actions.addWidget(self.save_button)

        content = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(content)
        layout.addLayout(header)
        layout.addWidget(self.inputs)
        layout.addWidget(self.error)
        layout.addLayout(actions)
        layout.addWidget(self.control)
        self.results = ResultView(self.form.describe, self)
        layout.addWidget(self.results, 1)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        self.setWidget(scroll)

    def _update_help(self):
        try:
            sampling = self._sampling()
        except ValueError as exc:
            text = str(exc)
        else:
            text = (
                f'{sampling.warmups} warmups; {sampling.rounds} rounds x '
                f'{sampling.repetitions} calls per kernel.')
        self.sampling_help.setText(text)

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
            result = results.load_artifact(path)
            self.results.set_result(result)
        except (OSError, ValueError) as exc:
            self.save_button.setEnabled(False)
            self._show_error(str(exc))
            return
        self.save_button.setEnabled(True)

    def _save(self):
        self._show_error('')
        filename = f'{self.form.OPERATION}-result.json'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save benchmark result', filename,
            'JSON files (*.json)')
        if not path:
            return
        try:
            results.write_artifact(results.load_artifact(self._path), path)
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


class MatmulForm:
    """Prepare Matmul controls within the inspector's shared form layout.

    Kernel choices and eligibility are ready after construction.
    :meth:`add_kernels` places them after the shared sampling controls.
    """

    LABEL = 'Matmul'
    OPERATION = matmul.MatmulSpec.OPERATION
    KERNEL_INFO = {
        'naive': ('Naive', 'Direct native loop for any valid input.'),
        'blas_dot': ('BLAS DOT', 'Vector @ vector.'),
        'blas_gevm': ('BLAS GEVM', 'Vector @ matrix.'),
        'blas_gemv': ('BLAS GEMV', 'Matrix @ vector.'),
        'blas_gemm': ('BLAS GEMM', 'Matrix @ matrix, including batches.'),
        'winograd': (
            'Winograd', 'Unbatched matrices with positive, even M, K and N.'),
    }

    def __init__(self, layout):
        parent = layout.parentWidget()
        self.dtype = QtWidgets.QComboBox(parent)
        self.dtype.addItems(matmul.MATMUL_DTYPES)
        self.dtype.setCurrentText('float64')
        self.dtype.setToolTip('Element type for both inputs and the result.')
        self.fields = {
            name: QtWidgets.QLineEdit(value, parent)
            for name, value in (
                ('lhs_shape', '64, 64'), ('lhs_strides', '64, 1'),
                ('rhs_shape', '64, 64'), ('rhs_strides', '64, 1'))
        }
        layout.addRow('Data type', self.dtype)
        operands = QtWidgets.QGridLayout()
        operands.addWidget(QtWidgets.QLabel('Shape'), 0, 1)
        operands.addWidget(QtWidgets.QLabel('Element strides'), 0, 2)
        for index, name in enumerate(('lhs', 'rhs'), 1):
            shape = self.fields[f'{name}_shape']
            strides = self.fields[f'{name}_strides']
            operands.addWidget(QtWidgets.QLabel('A' if index == 1 else 'B'),
                               index, 0)
            operands.addWidget(shape, index, 1)
            operands.addWidget(strides, index, 2)
            shape.setToolTip(
                'Full shape, including batch axes: e.g. 2, 32, 64.')
            strides.setToolTip(
                'One stride per axis, in elements. Negative and zero '
                'strides are allowed. Values are used exactly as entered.')
        operands.setColumnStretch(1, 1)
        operands.setColumnStretch(2, 1)
        layout.addRow('Operands', operands)
        self._build_kernels(parent)

    def make_spec(self, sampling):
        """Combine Matmul inputs with shared sampling."""
        inputs = self.make_inputs()
        kernels = tuple(name for name, box in self.kernels.items()
                        if box.isChecked())
        return matmul.MatmulSpec(
            lhs=inputs.lhs, rhs=inputs.rhs, dtype=inputs.dtype,
            sampling=sampling, kernels=kernels)

    def make_inputs(self):
        """Read exact operand metadata without preparing arrays."""
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

    def add_kernels(self, layout):
        """Place the prepared kernel choices after the sampling controls."""
        layout.addRow('Kernels', self._kernel_layout)

    @staticmethod
    def describe(specification):
        """Describe saved operands independently of the current controls."""
        lhs, rhs = specification.lhs, specification.rhs
        return (f'{specification.dtype}; '
                f'A {lhs.shape} strides {lhs.strides}; '
                f'B {rhs.shape} strides {rhs.strides}')

    def _build_kernels(self, parent):
        self._kernel_layout = QtWidgets.QGridLayout()
        self.kernels = {}
        self._kernel_choices = {}
        self._kernel_tooltips = {}
        for index, name in enumerate(matmul.MATMUL_KERNELS):
            label, tooltip = self.KERNEL_INFO[name]
            box = QtWidgets.QCheckBox(label, parent)
            if name != 'naive':
                tooltip += ' Requires a supported dtype and BLAS backend.'
            box.setToolTip(tooltip)
            box.setChecked(True)
            self._kernel_choices[name] = True
            self._kernel_tooltips[name] = tooltip
            box.toggled.connect(functools.partial(self._remember_kernel, name))
            self._kernel_layout.addWidget(box, index // 3, index % 3)
            self.kernels[name] = box
        self.dtype.currentTextChanged.connect(self._update_kernels)
        for field in self.fields.values():
            field.textChanged.connect(self._update_kernels)
        self._update_kernels()

    def _update_kernels(self):
        try:
            reasons = self.make_inputs().kernel_eligibility()
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


class ResultView(QtWidgets.QGroupBox):
    """Display completed results independently of the current controls.

    :param describe: Format the saved spec as summary text, without reading
        live input controls.
    """

    HEADERS = ('Kernel', 'Median (ns/call)', 'p95 (ns/call)',
               'Max abs diff', 'Relative diff', 'Status')

    def __init__(self, describe, parent=None):
        super().__init__('Results', parent)
        self._describe = describe
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
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().hide()
        header = self.table.horizontalHeader()
        mode = QtWidgets.QHeaderView.ResizeMode
        header.setSectionResizeMode(mode.ResizeToContents)
        header.setMinimumSectionSize(header.sectionSizeHint(1))
        header.setDefaultSectionSize(header.minimumSectionSize())
        header.setSectionResizeMode(1, mode.Stretch)
        header.setSectionResizeMode(2, mode.Stretch)
        self.table.setMinimumHeight(140)
        align = QtCore.Qt.AlignmentFlag
        header.setDefaultAlignment(align.AlignLeft | align.AlignVCenter)
        for column in range(1, 5):
            self.table.horizontalHeaderItem(column).setTextAlignment(
                align.AlignRight | align.AlignVCenter)
        self.table.horizontalHeaderItem(3).setToolTip(
            'Maximum absolute difference from the NumPy result.')
        self.table.horizontalHeaderItem(4).setToolTip(
            'Max abs diff / max(abs(NumPy result)); "-" means unavailable.')
        legend = QtWidgets.QLabel(
            'Bar: median  |  Range: p5-p95  |  Lower is faster', self)
        legend.setToolTip(
            'Percentiles use per-call round averages. '
            'Few rounds give coarse tail estimates. Lower is faster. '
            'Hover over a kernel for timings, differences, and sampling.')
        legend.setWordWrap(True)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        splitter.addWidget(self.chart)
        splitter.addWidget(self.table)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([220, 180])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(legend)
        layout.addWidget(splitter, 1)

    def set_result(self, result):
        """Display a completed :class:`~solvcon.benchmark.results.RunResult`.

        :param result: Validated run, with its saved spec and raw timings.
        """
        rows = self._make_rows(result)
        maximum = max((row['timing'].p95 or 0 for row in rows), default=0)
        scale, unit = 1, 'ns'
        for factor, name in ((1e3, '\u00b5s'), (1e6, 'ms'), (1e9, 's')):
            if maximum >= factor:
                scale, unit = factor, name
        for column, label in ((1, 'Median'), (2, 'p95')):
            self.table.horizontalHeaderItem(column).setText(
                f'{label} ({unit}/call)')
        align = QtCore.Qt.AlignmentFlag
        sampling = result.spec.sampling
        self.summary.setText(
            self._describe(result.spec) + '\n'
            f'{sampling.rounds} round averages; '
            f'{sampling.repetitions} calls/round')
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            timing = row['timing']
            values = (row['label'],
                      _format_number(timing.median, scale),
                      _format_number(timing.p95, scale),
                      _format_number(row['max_abs_diff']),
                      _format_number(row['relative_diff']),
                      row['status'])
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(row['tooltip'])
                if 1 <= column <= 4:
                    item.setTextAlignment(
                        align.AlignRight | align.AlignVCenter)
                self.table.setItem(index, column, item)
        measured = [row for row in rows if row['status'] == 'measured']
        self.chart.set_rows(measured, scale, unit)

    @staticmethod
    def _make_rows(result):
        sampling = result.spec.sampling
        repetitions = sampling.repetitions
        timings = result.timing_stats()
        rows = []
        for entry in result.results:
            timing = timings[entry.name]
            row = dict(entry.to_dict(), timing=timing)
            row['label'] = 'NumPy' if entry.name == 'numpy' else entry.name
            row['tooltip'] = (
                f'{entry.name}: {entry.status}\n'
                f'Median: {_format_number(timing.median)} ns/call\n'
                f'p5: {_format_number(timing.p5)} ns/call\n'
                f'p25: {_format_number(timing.p25)} ns/call\n'
                f'p75: {_format_number(timing.p75)} ns/call\n'
                f'p95: {_format_number(timing.p95)} ns/call\n'
                f'Max abs diff: {_format_number(entry.max_abs_diff)}\n'
                f'Relative diff: {_format_number(entry.relative_diff)}\n'
                f'{sampling.rounds} rounds, {repetitions} calls/round; '
                f'{sampling.warmups} warmups\n'
                'Percentiles use per-call round averages.'
            )
            if entry.reason:
                row['tooltip'] += f'\n{entry.reason}'
            rows.append(row)
        return rows


class TimingChart(QtWidgets.QWidget):
    """Show median bars and p5-p95 ranges of per-call round averages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._regions = []
        self._scale, self._unit = 1, 'ns'
        self._empty_text = 'Run a benchmark to compare kernel timings'
        self.setMouseTracking(True)
        self.setMinimumSize(420, 180)

    def set_rows(self, rows, scale=1, unit='ns'):
        self._rows = rows
        self._scale, self._unit = scale, unit
        self._regions = []
        self._empty_text = 'No measured timings in the completed result'
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()
        self.update()

    @staticmethod
    def _ticks(maximum):
        """Return ticks with 1, 2 or 5 times a power-of-ten spacing.

        :param maximum: Largest percentile in the displayed time unit.
        """
        maximum = maximum or 1
        target = maximum / 5
        step = 10 ** math.floor(math.log10(target))
        for factor in (1, 2, 5, 10):
            if step * factor >= target:
                step *= factor
                break
        intervals = math.ceil(maximum / step)
        return [index * step for index in range(intervals + 1)]

    def _draw_axis(self, painter, plot, ticks):
        metrics = self.fontMetrics()
        color = self.palette().text().color()
        grid = QtGui.QColor(color)
        grid.setAlpha(30)

        for value in ticks:
            position = plot.left() + plot.width() * value / ticks[-1]
            painter.setPen(grid)
            painter.drawLine(QtCore.QPointF(position, plot.top()),
                             QtCore.QPointF(position, plot.bottom()))
            painter.setPen(color)
            painter.drawLine(QtCore.QPointF(position, plot.bottom()),
                             QtCore.QPointF(position, plot.bottom() + 4))
            text = _format_number(value)
            width = metrics.horizontalAdvance(text) + 8
            rect = QtCore.QRectF(position - width / 2, plot.bottom() + 4,
                                 width, metrics.height())
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())

    def paintEvent(self, event):
        self._regions = []
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        align = QtCore.Qt.AlignmentFlag
        if not self._rows:
            painter.drawText(self.rect(), align.AlignCenter, self._empty_text)
            return

        maximum = max(row['timing'].p95 for row in self._rows)
        ticks = self._ticks(maximum / self._scale)
        axis_max = ticks[-1] * self._scale

        metrics = self.fontMetrics()
        label_width = max(metrics.horizontalAdvance(row['label'])
                          for row in self._rows)
        value_header = f'Median ({self._unit}/call)'
        values = [_format_number(row['timing'].median, self._scale)
                  for row in self._rows]
        value_width = max(metrics.horizontalAdvance(text)
                          for text in [value_header, *values])
        padding, gap = 8, 12
        top, axis_height = metrics.height() + 16, metrics.height() + 8
        left = padding + label_width + gap
        right = gap + value_width + padding
        plot = QtCore.QRectF(
            left, top, self.width() - left - right,
            self.height() - top - axis_height - padding,
        )

        painter.drawText(
            QtCore.QRectF(plot.left(), 0, plot.width(), top),
            align.AlignLeft | align.AlignVCenter, f'Time ({self._unit}/call)')
        painter.drawText(
            QtCore.QRectF(plot.right() + gap, 0, value_width, top),
            align.AlignRight | align.AlignVCenter, value_header)

        height = plot.height() / len(self._rows)
        for index, row in enumerate(self._rows):
            rect = QtCore.QRectF(0, plot.top() + index * height,
                                 self.width(), height)
            self._regions.append((rect, row['tooltip']))
        for rect, _ in self._regions[1::2]:
            painter.fillRect(rect, self.palette().alternateBase())
        self._draw_axis(painter, plot, ticks)

        for index, row in enumerate(self._rows):
            timing = row['timing']
            rect = self._regions[index][0]
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QtCore.QRectF(padding, rect.top(), label_width, height),
                align.AlignRight | align.AlignVCenter, row['label'],
            )
            middle = rect.center().y()
            bar_height = min(20, height * 0.6)
            p5_x, median_x, p95_x = (
                plot.left() + plot.width() * value / axis_max
                for value in (timing.p5, timing.median, timing.p95))
            color = self.palette().highlight().color()
            color.setAlpha(180)
            bar = QtCore.QRectF(
                plot.left(), middle - bar_height / 2,
                median_x - plot.left(), bar_height)
            painter.fillRect(bar, color)
            painter.drawLine(QtCore.QLineF(p5_x, middle, p95_x, middle))
            y1, y2 = middle - bar_height / 3, middle + bar_height / 3
            painter.drawLine(QtCore.QLineF(p5_x, y1, p5_x, y2))
            painter.drawLine(QtCore.QLineF(p95_x, y1, p95_x, y2))
            painter.drawText(
                QtCore.QRectF(
                    plot.right() + gap, rect.top(), value_width, height,
                ),
                align.AlignRight | align.AlignVCenter,
                values[index],
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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
