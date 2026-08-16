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

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QGridLayout, QLabel, QComboBox, QDoubleSpinBox,
                               QSpinBox, QPushButton, QToolButton, QSizePolicy,
                               QScrollArea, QFrame)

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


def _count(value, low, high):
    box = QSpinBox()
    box.setRange(low, high)
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


def _reserve_width(button, texts):
    """Reserve the width of the widest label a button swaps through, so the
    swap cannot resize the button and jitter the panel around it."""
    keep = button.text()
    width = 0
    for text in texts:
        button.setText(text)
        width = max(width, button.sizeHint().width())
    button.setText(keep)
    button.setMinimumWidth(width)


class FoldBox(QWidget):
    """A titled section that folds behind an arrow header.

    Qt ships no folding section, so the header is the usual flat arrow
    tool button over a content pane whose visibility it toggles; a folded
    section gives its room back to the boxes below it.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._head = QToolButton()
        self._head.setText(title)
        self._head.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._head.setArrowType(Qt.DownArrow)
        self._head.setAutoRaise(True)
        # Span the panel, so the whole header line takes the click.  The
        # fold state lives here, not in a checkable button: a checked
        # tool button draws pressed, which would box every open header.
        self._head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font = self._head.font()
        font.setBold(True)
        self._head.setFont(font)
        # Hold the header to the label's own height: the arrow follows the
        # font instead of the default tool-button icon, whose padding made
        # every header taller than its text.
        metrics = self._head.fontMetrics()
        self._head.setIconSize(QSize(metrics.ascent(), metrics.ascent()))
        self._head.setFixedHeight(metrics.height())
        self._open = True
        self._head.clicked.connect(self._on_head_clicked)
        self._content = QWidget()

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self._head)
        box.addWidget(self._content)

    def _on_head_clicked(self):
        self._open = not self._open
        self._head.setArrowType(Qt.DownArrow if self._open else Qt.RightArrow)
        self._content.setVisible(self._open)

    # A hidden widget drops out of the layout's size negotiation, so a
    # fold would narrow the box to its header and shift the panel width
    # with it.  Folding may only give back height: both hints keep
    # answering for the content's width while it is hidden.

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(max(hint.width(), self._content.sizeHint().width()))
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(max(hint.width(),
                          self._content.minimumSizeHint().width()))
        return hint


class FreeStreamBox(FoldBox):
    """The upstream state and the incident shock; read once at Start."""

    def __init__(self, parent=None):
        super().__init__("Free stream", parent)
        self._gamma = _spin(1.4, 1.01, 3.0, 0.01, 3)
        self._density = _spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._pressure = _spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._mach = _spin(3.0, 1.1, 20.0, 0.1, 3)
        self._angle = _spin(10.0, 0.5, 45.0, 0.5, 2)
        self._angle.setSuffix(" deg")

        form = QFormLayout(self._content)
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


class NumericsBox(FoldBox):
    """The discretization of a run: how finely the domain is cut and how far
    a step carries it; read at Start and at Remesh.

    The mesh is ``nx`` by ``ny`` boxes over a domain four units long and one
    tall, so ``nx = 4 * ny`` keeps the cells square.  A finer mesh also holds
    a shorter stable step, which the time step has to follow.
    """

    #: Mesh flavors offered by :mod:`._driver`, the first being the default.
    CELL_TYPES = ('unstructured', 'quad', 'triangle')

    def __init__(self, parent=None):
        super().__init__("Numerics", parent)
        # Owner-supplied callback that rebuilds the run.
        self.remesh_requested = None
        self._nx = _count(64, 4, 1024)
        self._ny = _count(16, 2, 256)
        self._dt = _spin(2e-3, 1e-6, 1.0, 1e-3, 6)
        self._cell_type = QComboBox()
        self._cell_type.addItems(self.CELL_TYPES)
        # Apply a new resolution without marching it.
        self._remesh = QPushButton("Remesh")
        self._remesh.clicked.connect(self._on_remesh_clicked)

        form = QFormLayout(self._content)
        form.addRow("nx", self._nx)
        form.addRow("ny", self._ny)
        form.addRow("time step", self._dt)
        form.addRow("cell type", self._cell_type)
        form.addRow(self._remesh)

    def params(self):
        return dict(nx=self._nx.value(), ny=self._ny.value(),
                    time_increment=self._dt.value(),
                    cell_type=self._cell_type.currentText())

    def _on_remesh_clicked(self):
        if self.remesh_requested is not None:
            self.remesh_requested()


class RunBox(FoldBox):
    """The march controls and the live march readout, kept side by side.

    The buttons stand in the order a run is used: started, held (paused,
    stepped), and ended (stopped where it stands, or dropped).  The readout
    beneath them carries the step count, what ended the run, and the mass
    the domain holds.
    """

    #: What ended a run reads as in the state cell; a live run reads as
    #: "running" or "paused" from the Pause button instead.
    STOP_STATES = {'cap': "step cap", 'stopped': "stopped"}

    def __init__(self, parent=None):
        super().__init__("Run", parent)
        # Owner-supplied callbacks that drive the solver from the controls.
        self.viewer_toggled = None
        self.start_requested = None
        self.pause_toggled = None
        self.step_requested = None
        self.stop_requested = None
        self.reset_requested = None
        self._paused = False
        self._live = False

        self._steps = _count(5, 1, 1000)

        # Opens and closes the one domain viewer the run buttons draw into.
        self._viewer_btn = QPushButton("Open viewer")
        self._viewer_btn.setCheckable(True)
        self._viewer_btn.toggled.connect(self._on_viewer_toggled)
        _reserve_width(self._viewer_btn, ("Open viewer", "Close viewer"))

        self._start = QPushButton("Start")
        self._start.clicked.connect(self._on_start_clicked)
        self._pause = QPushButton("Pause")
        self._pause.setCheckable(True)
        self._pause.toggled.connect(self._on_pause_toggled)
        self._step = QPushButton("Step")
        self._step.clicked.connect(self._on_step_clicked)
        _reserve_width(self._pause, ("Pause", "Resume"))
        self._stop = QPushButton("Stop")
        self._stop.clicked.connect(self._on_stop_clicked)
        self._reset = QPushButton("Reset")
        self._reset.clicked.connect(self._on_reset_clicked)
        marching = QHBoxLayout()
        marching.addWidget(self._start)
        marching.addWidget(self._pause)
        marching.addWidget(self._step)
        ending = QHBoxLayout()
        ending.addWidget(self._stop)
        ending.addWidget(self._reset)

        self._progress = _value_label()
        self._state = _value_label()
        self._mass = _value_label()

        form = QFormLayout(self._content)
        form.addRow("steps/frame", self._steps)
        form.addRow(self._viewer_btn)
        form.addRow(marching)
        form.addRow(ending)
        form.addRow("step", self._progress)
        form.addRow("state", self._state)
        form.addRow("mass", self._mass)
        self.show_run(None)

    def steps_per_frame(self):
        return self._steps.value()

    def set_paused(self, paused):
        """Reflect the run state in the Pause button without re-firing it."""
        self._pause.blockSignals(True)
        self._pause.setChecked(paused)
        self._pause.blockSignals(False)
        self._show_paused(paused)

    def _show_paused(self, paused):
        """Carry the pause into the button text and, while a live run is
        showing, into the state cell, which no frame redraws on a pause
        because pausing stops the frames."""
        self._paused = paused
        self._pause.setText("Resume" if paused else "Pause")
        if self._live:
            self._state.setText("paused" if paused else "running")

    def set_viewer_open(self, open_):
        """Reflect the viewer state in its button without re-firing it."""
        self._viewer_btn.blockSignals(True)
        self._viewer_btn.setChecked(open_)
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        self._viewer_btn.blockSignals(False)

    def show_run(self, session):
        """Read the march progress of one run, or the lack of a run."""
        self._live = session is not None and session.stop_reason is None
        # A run that has ended can still be dropped, but not marched.
        for button in (self._pause, self._step, self._stop):
            button.setEnabled(self._live)
        self._reset.setEnabled(session is not None)
        if None is session:
            self._progress.setText("-")
            self._state.setText("not started")
            self._mass.setText("-")
            return
        if self._live:
            state = "paused" if self._paused else "running"
        else:
            state = self.STOP_STATES[session.stop_reason]
        self._progress.setText(f"{session.step} / {session.max_steps}")
        self._state.setText(state)
        last = session.history.last
        self._mass.setText("-" if last is None else _number(last.mass))

    def _on_viewer_toggled(self, open_):
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        if self.viewer_toggled is not None:
            self.viewer_toggled(open_)

    def _on_start_clicked(self):
        if self.start_requested is not None:
            self.start_requested()

    def _on_pause_toggled(self, paused):
        self._show_paused(paused)
        if self.pause_toggled is not None:
            self.pause_toggled(paused)

    def _on_step_clicked(self):
        if self.step_requested is not None:
            self.step_requested()

    def _on_stop_clicked(self):
        if self.stop_requested is not None:
            self.stop_requested()

    def _on_reset_clicked(self):
        if self.reset_requested is not None:
            self.reset_requested()


class FieldBox(FoldBox):
    """Which derived field the viewer colors, and the range it spans."""

    #: Derived scalar fields the viewer can color, in display order.
    FIELDS = EulerField.FIELDS

    def __init__(self, parent=None):
        super().__init__("Field", parent)
        self.field_changed = None
        self._selector = QComboBox()
        self._selector.addItems(self.FIELDS)
        self._selector.currentTextChanged.connect(self._on_field_changed)
        self._min = _value_label()
        self._max = _value_label()

        form = QFormLayout(self._content)
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


class ZoneBox(FoldBox):
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
        self._grid = QGridLayout(self._content)
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


class SolutionPanel(QScrollArea):
    """Widget with the solver controls and a live solution-field readout.

    The panel is pure composition: it stacks the boxes and forwards their
    callbacks and accessors as one flat surface, so the controller and the
    tests see a single widget while every box stays its own concern.  The
    stack scrolls vertically when the dock stands shorter than the boxes,
    so nothing is ever cut off; it never scrolls sideways, and the dock
    cannot be squeezed below the width the form needs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freestream = FreeStreamBox()
        self._numerics = NumericsBox()
        self._run = RunBox()
        self._field = FieldBox()
        self._zones = ZoneBox()
        self._boxes = (self._freestream, self._numerics, self._run,
                       self._field, self._zones)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        for box in self._boxes:
            layout.addWidget(box)
        layout.addStretch(1)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(inner.minimumSizeHint().width())
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
    def stop_requested(self):
        return self._run.stop_requested

    @stop_requested.setter
    def stop_requested(self, callback):
        self._run.stop_requested = callback

    @property
    def reset_requested(self):
        return self._run.reset_requested

    @reset_requested.setter
    def reset_requested(self, callback):
        self._run.reset_requested = callback

    @property
    def remesh_requested(self):
        return self._numerics.remesh_requested

    @remesh_requested.setter
    def remesh_requested(self, callback):
        self._numerics.remesh_requested = callback

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
