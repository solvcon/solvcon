# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Control widget for the oblique-shock reflection run.

The input boxes hold what a run consumes when it starts: the free stream
and the incident shock in one, the discretization in the other.  The run
box holds the march controls.  Below them the field selector picks what
the viewer colors, and the status tree reads out the run.  Each box is a
passive widget that only reports what its controls hold and calls back
into its owner; the panel composes the boxes and forwards their surface,
and the solver and the viewer belong to the controller in
:mod:`._control`.
"""

import math

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QComboBox, QDoubleSpinBox, QSpinBox,
                               QPushButton, QTreeWidget, QTreeWidgetItem,
                               QFrame, QGroupBox)

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
    """The march controls: the pacing, the viewer, and the run buttons."""

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
        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._pause)
        buttons.addWidget(self._step)

        form = QFormLayout(self)
        form.addRow("steps/frame", self._steps)
        form.addRow(self._viewer_btn)
        form.addRow(buttons)

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


class SolutionPanel(QWidget):
    """Widget with the solver controls and a live solution-field readout.

    The panel composes the input and run boxes and forwards their
    callbacks and accessors as one flat surface, so the controller and
    the tests see a single widget.  The field selector and the status
    tree stay on the panel itself until the readout grows boxes of its
    own.
    """

    #: Derived scalar fields the viewer can color, in display order.
    FIELDS = EulerField.FIELDS
    #: What a :attr:`ReflectionSession.stop_reason` reads as in the tree.
    RUN_STATES = {None: "running", 'cap': "step cap", 'stopped': "stopped"}
    #: Headings of the status tree, whose first column carries the labels.
    STATUS_COLUMNS = ("", "value", "analytic", "error")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_changed = None
        self._freestream = FreeStreamBox()
        self._numerics = NumericsBox()
        self._run = RunBox()
        self._field = QComboBox()
        self._field.addItems(self.FIELDS)
        self._field.currentTextChanged.connect(self._on_field_changed)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.addWidget(self._freestream)
        self._layout.addWidget(self._numerics)
        self._layout.addWidget(self._run)
        form = QFormLayout()
        form.addRow("field", self._field)
        self._layout.addLayout(form)
        self._build_status()

    def _build_status(self):
        """Add the read-only run readout below the controls.

        The headings are what keep a zone's three numbers apart; only the
        zone rows fill the last two columns, and the rest read as a plain
        label / value list under them.
        """
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(self.STATUS_COLUMNS)
        self._tree.setFrameShape(QFrame.NoFrame)
        self._layout.addWidget(self._tree)
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

    def params(self):
        """Collect the current control values for the solver driver."""
        return {**self._freestream.params(), **self._numerics.params()}

    def field(self):
        return self._field.currentText()

    def steps_per_frame(self):
        return self._run.steps_per_frame()

    def set_paused(self, paused):
        self._run.set_paused(paused)

    def set_viewer_open(self, open_):
        self._run.set_viewer_open(open_)

    def set_status(self, session, vmin, vmax):
        """Read one run out into the status tree.

        ``session`` is the :class:`~._session.ReflectionSession` being
        marched, or None before the first run; ``vmin`` and ``vmax`` bound
        the field the viewer is drawing.

        The zone rows are what judge the run.  Each carries its analytic
        value beside the computed one, so the error in the last column is a
        reading a user can check rather than take on faith, and the analytic
        state is the only thing the field is measured against: how far a
        march has come is the step count, not the size of what it last
        changed.
        """
        self._tree.clear()
        if None is session:
            QTreeWidgetItem(self._tree, ["not started"])
            return
        name = self.field()
        self._row("step", f"{session.step}")
        self._row("run", self.RUN_STATES[session.stop_reason])
        self._row("field", name)
        self._row("min", self._number(vmin))
        self._row("max", self._number(vmax))
        for info in session.zone_info(name):
            # A zone whose analytic value is zero, as the transverse
            # velocity is in zones 1 and 3, has no error to be a percent of.
            error = "" if math.isnan(info.error) else f"{info.error:+.2%}"
            self._row(f"zone {info.zone}", self._number(info.computed),
                      self._number(info.analytic), error)
        for column in range(self._tree.columnCount() - 1):
            self._tree.resizeColumnToContents(column)

    @staticmethod
    def _number(value):
        """Format a readout number to four significant digits.

        The alternate form keeps the trailing zeros a plain ``g`` drops, so
        a column of values stays a column instead of ragged decimals.
        """
        return f"{value:#.4g}"

    def _row(self, label, value, analytic="", error=""):
        """Add one line to the status tree, under its column headings."""
        QTreeWidgetItem(self._tree, [label, value, analytic, error])

    def _on_field_changed(self, name):
        if self.field_changed is not None:
            self.field_changed(name)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
