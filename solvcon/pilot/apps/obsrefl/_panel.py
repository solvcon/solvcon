# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Control widget for the oblique-shock reflection run.

The panel stacks its boxes in the order a run is used: the free-stream and
numerics inputs consumed when a run starts, the run controls with the live
march readout, the field the viewer colors with the range it spans, and
the zone readout that judges the field against the analytic reflection.
Each box is a passive widget that only reports what its controls hold and
calls back into its owner; the panel composes the boxes and forwards their
surface, and the solver and the viewer belong to the controller in
:mod:`._control`.
"""

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QGridLayout, QGroupBox, QLabel, QComboBox,
                               QDoubleSpinBox, QSpinBox, QPushButton)

from ....multidim.euler import EulerField

__all__ = [  # noqa: F822
    'SolutionPanel',
]


def _spin(value, low, high, step, decimals):
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _number(value):
    """Format a readout number to four significant digits.

    The alternate form keeps the trailing zeros a plain ``g`` drops, so a
    column of values stays a column instead of ragged decimals.
    """
    return f"{value:#.4g}"


def _value_label():
    """A live readout cell: right-aligned in a fixed-width font, so a value
    can update every frame without jittering the column it stands in."""
    label = QLabel("")
    label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    return label


class FreeStreamBox(QGroupBox):
    """The upstream state and the incident shock; read once at Start."""

    def __init__(self, parent=None):
        super().__init__("Free stream", parent)
        self.setFlat(True)
        self._gamma = _spin(1.4, 1.01, 3.0, 0.01, 3)
        self._density = _spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._pressure = _spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._mach = _spin(3.0, 1.1, 20.0, 0.1, 3)
        self._angle = _spin(10.0, 0.5, 45.0, 0.5, 2)
        self._angle.setSuffix(" deg")

        form = QFormLayout(self)
        form.addRow("gamma", self._gamma)
        form.addRow("density", self._density)
        form.addRow("pressure", self._pressure)
        form.addRow("Mach", self._mach)
        form.addRow("shock angle", self._angle)

    def params(self):
        return dict(gamma=self._gamma.value(),
                    density=self._density.value(),
                    pressure=self._pressure.value(),
                    mach=self._mach.value(),
                    angle=self._angle.value())


class NumericsBox(QGroupBox):
    """The discretization of a run; read once at Start."""

    #: Mesh flavors offered by :mod:`._driver`, the first being the default.
    CELL_TYPES = ('unstructured', 'quad', 'triangle')

    def __init__(self, parent=None):
        super().__init__("Numerics", parent)
        self.setFlat(True)
        self._dt = _spin(2e-3, 1e-6, 1.0, 1e-3, 6)
        self._cell_type = QComboBox()
        self._cell_type.addItems(self.CELL_TYPES)

        form = QFormLayout(self)
        form.addRow("time step", self._dt)
        form.addRow("cell type", self._cell_type)

    def params(self):
        return dict(time_increment=self._dt.value(),
                    cell_type=self._cell_type.currentText())


class RunBox(QGroupBox):
    """The march controls and the live march readout, kept side by side.

    The readout is the run's own progress: how far the march has come of
    the steps it may take, what ended it, and the overall mass the domain
    holds, which the inflow and the outflow move as the flow develops.
    """

    #: What a :attr:`ReflectionSession.stop_reason` reads as in the readout.
    RUN_STATES = {None: "running", 'cap': "step cap", 'stopped': "stopped"}

    def __init__(self, parent=None):
        super().__init__("Run", parent)
        self.setFlat(True)
        # Owner-supplied callbacks that drive the solver from the controls.
        self.viewer_toggled = None
        self.start_requested = None
        self.pause_toggled = None
        self.step_requested = None

        self._steps = QSpinBox()
        self._steps.setRange(1, 1000)
        self._steps.setValue(5)

        # Opens and closes the one domain viewer the run buttons draw into.
        self._viewer_btn = QPushButton("Open viewer")
        self._viewer_btn.setCheckable(True)
        self._viewer_btn.toggled.connect(self._on_viewer_toggled)

        self._start = QPushButton("Start")
        self._start.clicked.connect(self._on_start_clicked)
        self._pause = QPushButton("Pause")
        self._pause.setCheckable(True)
        self._pause.toggled.connect(self._on_pause_toggled)
        self._step = QPushButton("Step")
        self._step.clicked.connect(self._on_step_clicked)
        # Disabled, not hidden, until a run exists to pause or step.
        self._pause.setEnabled(False)
        self._step.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._pause)
        buttons.addWidget(self._step)

        self._progress = _value_label()
        self._state = _value_label()
        self._mass = _value_label()

        form = QFormLayout(self)
        form.addRow("steps/frame", self._steps)
        form.addRow(self._viewer_btn)
        form.addRow(buttons)
        form.addRow("step", self._progress)
        form.addRow("state", self._state)
        form.addRow("mass", self._mass)

    def steps_per_frame(self):
        return self._steps.value()

    def set_paused(self, paused):
        """Reflect the run state in the Pause button without re-firing it."""
        self._pause.blockSignals(True)
        self._pause.setChecked(paused)
        self._pause.setText("Resume" if paused else "Pause")
        self._pause.blockSignals(False)

    def set_viewer_open(self, open_):
        """Reflect the viewer state in its button without re-firing it."""
        self._viewer_btn.blockSignals(True)
        self._viewer_btn.setChecked(open_)
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        self._viewer_btn.blockSignals(False)

    def show_run(self, session):
        """Read the march progress of one run, or the lack of a run."""
        if None is session:
            self._progress.setText("-")
            self._state.setText("not started")
            self._mass.setText("-")
            return
        self._progress.setText(f"{session.step} / {session.max_steps}")
        self._state.setText(self.RUN_STATES[session.stop_reason])
        last = session.history.last
        self._mass.setText("-" if last is None else _number(last.mass))
        self._pause.setEnabled(True)
        self._step.setEnabled(True)

    def _on_viewer_toggled(self, open_):
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        if self.viewer_toggled is not None:
            self.viewer_toggled(open_)

    def _on_start_clicked(self):
        if self.start_requested is not None:
            self.start_requested()

    def _on_pause_toggled(self, paused):
        self._pause.setText("Resume" if paused else "Pause")
        if self.pause_toggled is not None:
            self.pause_toggled(paused)

    def _on_step_clicked(self):
        if self.step_requested is not None:
            self.step_requested()


class FieldBox(QGroupBox):
    """Which derived field the viewer colors, and the range it spans."""

    #: Derived scalar fields the viewer can color, in display order.
    FIELDS = EulerField.FIELDS

    def __init__(self, parent=None):
        super().__init__("Field", parent)
        self.setFlat(True)
        self.field_changed = None
        self._selector = QComboBox()
        self._selector.addItems(self.FIELDS)
        self._selector.currentTextChanged.connect(self._on_field_changed)
        self._min = _value_label()
        self._max = _value_label()

        form = QFormLayout(self)
        form.addRow("field", self._selector)
        form.addRow("min", self._min)
        form.addRow("max", self._max)

    def field(self):
        return self._selector.currentText()

    def show_range(self, vmin, vmax):
        self._min.setText("-" if vmin is None else _number(vmin))
        self._max.setText("-" if vmax is None else _number(vmax))

    def _on_field_changed(self, name):
        if self.field_changed is not None:
            self.field_changed(name)


class ZoneBox(QGroupBox):
    """The per-zone readout that judges the run.

    Each zone carries its analytic value beside the computed one, so the
    error in the last column is a reading a user can check rather than take
    on faith, and the analytic state is the only thing the field is
    measured against: how far a march has come is the step count, not the
    size of what it last changed.
    """

    HEADERS = ("zone", "computed", "analytic", "error")

    def __init__(self, parent=None):
        super().__init__("Zones", parent)
        self.setFlat(True)
        self._grid = QGridLayout(self)
        for col, text in enumerate(self.HEADERS):
            label = QLabel(text)
            font = label.font()
            font.setBold(True)
            label.setFont(font)
            if col:
                label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._grid.addWidget(label, 0, col)
        self._rows = []

    def show_zones(self, infos):
        """Fill one row per zone; a missing run blanks the rows instead of
        removing them, so the panel does not jump as a run comes and goes."""
        while len(self._rows) < len(infos):
            row = tuple(_value_label() for _ in range(len(self.HEADERS)))
            for col, label in enumerate(row):
                self._grid.addWidget(label, len(self._rows) + 1, col)
            self._rows.append(row)
        for row in self._rows:
            for label in row:
                label.setText("")
        for info, row in zip(infos, self._rows):
            row[0].setText(f"{info.zone}")
            row[1].setText(_number(info.computed))
            row[2].setText(_number(info.analytic))
            # A zone whose analytic value is zero, as the transverse
            # velocity is in zones 1 and 3, has no error to be a percent of.
            error = "" if math.isnan(info.error) else f"{info.error:+.2%}"
            row[3].setText(error)


class SolutionPanel(QWidget):
    """Widget with the solver controls and a live solution-field readout.

    The panel is pure composition: it stacks the boxes and forwards their
    callbacks and accessors as one flat surface, so the controller and the
    tests see a single widget while every box stays its own concern.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freestream = FreeStreamBox()
        self._numerics = NumericsBox()
        self._run = RunBox()
        self._field = FieldBox()
        self._zones = ZoneBox()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        for box in (self._freestream, self._numerics, self._run, self._field,
                    self._zones):
            layout.addWidget(box)
        layout.addStretch(1)
        self.set_status(None, None, None)

    @property
    def viewer_toggled(self):
        return self._run.viewer_toggled

    @viewer_toggled.setter
    def viewer_toggled(self, callback):
        self._run.viewer_toggled = callback

    @property
    def start_requested(self):
        return self._run.start_requested

    @start_requested.setter
    def start_requested(self, callback):
        self._run.start_requested = callback

    @property
    def pause_toggled(self):
        return self._run.pause_toggled

    @pause_toggled.setter
    def pause_toggled(self, callback):
        self._run.pause_toggled = callback

    @property
    def step_requested(self):
        return self._run.step_requested

    @step_requested.setter
    def step_requested(self, callback):
        self._run.step_requested = callback

    @property
    def field_changed(self):
        return self._field.field_changed

    @field_changed.setter
    def field_changed(self, callback):
        self._field.field_changed = callback

    def params(self):
        """Collect the current control values for the solver driver."""
        return {**self._freestream.params(), **self._numerics.params()}

    def field(self):
        return self._field.field()

    def steps_per_frame(self):
        return self._run.steps_per_frame()

    def set_paused(self, paused):
        self._run.set_paused(paused)

    def set_viewer_open(self, open_):
        self._run.set_viewer_open(open_)

    def set_status(self, session, vmin, vmax):
        """Read one run out into the readout boxes.

        ``session`` is the :class:`~._session.ReflectionSession` being
        marched, or None before the first run; ``vmin`` and ``vmax`` bound
        the field the viewer is drawing.
        """
        self._run.show_run(session)
        if None is session:
            self._field.show_range(None, None)
            self._zones.show_zones([])
            return
        self._field.show_range(vmin, vmax)
        self._zones.show_zones(session.zone_info(self.field()))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
