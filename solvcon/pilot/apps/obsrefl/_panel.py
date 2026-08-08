# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Control widget for the oblique-shock reflection run.

The controls set the free stream and the mesh, open or close the domain
viewer sub-window, start / pause / step the march, and pick which derived
field (density, velocity, pressure, Mach, ...) the viewer colors.  Below
them the status tree reads out the run: how far it has come, and how far
each zone stands from the analytic state it has to hold.  The widget only
reports what its controls hold and calls back into its owner; the solver
and the viewer belong to the feature in :mod:`._app`.
"""

import math

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QComboBox, QDoubleSpinBox, QSpinBox,
                               QPushButton, QTreeWidget, QTreeWidgetItem,
                               QFrame)

from ....multidim.euler import EulerField

__all__ = [  # noqa: F822
    'SolutionPanel',
]


class SolutionPanel(QWidget):
    """Widget with the solver controls and a live solution-field readout."""

    #: Derived scalar fields the viewer can color, in display order.
    FIELDS = EulerField.FIELDS
    #: Mesh flavors offered by :mod:`._driver`, the first being the default.
    CELL_TYPES = ('unstructured', 'quad', 'triangle')
    #: What a :attr:`ReflectionSession.stop_reason` reads as in the tree.
    RUN_STATES = {None: "running", 'cap': "step cap", 'stopped': "stopped"}
    #: Headings of the status tree, whose first column carries the labels.
    STATUS_COLUMNS = ("", "value", "analytic", "error")

    def __init__(self, parent=None):
        super().__init__(parent)
        # Owner-supplied callbacks that drive the solver from the controls.
        self.viewer_toggled = None
        self.start_requested = None
        self.pause_toggled = None
        self.step_requested = None
        self.field_changed = None
        self._build_controls()
        self._build_status()

    def _build_controls(self):
        """Lay out the free-stream / mesh inputs and the run buttons."""
        self._gamma = self._spin(1.4, 1.01, 3.0, 0.01, 3)
        self._density = self._spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._pressure = self._spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._mach = self._spin(3.0, 1.1, 20.0, 0.1, 3)
        self._angle = self._spin(10.0, 0.5, 45.0, 0.5, 2)
        self._dt = self._spin(2e-3, 1e-6, 1.0, 1e-3, 6)
        self._steps = QSpinBox()
        self._steps.setRange(1, 1000)
        self._steps.setValue(5)
        self._cell_type = QComboBox()
        self._cell_type.addItems(self.CELL_TYPES)
        self._field = QComboBox()
        self._field.addItems(self.FIELDS)
        self._field.currentTextChanged.connect(self._on_field_changed)

        form = QFormLayout()
        form.addRow("gamma", self._gamma)
        form.addRow("density", self._density)
        form.addRow("pressure", self._pressure)
        form.addRow("mach", self._mach)
        form.addRow("angle (deg)", self._angle)
        form.addRow("dt", self._dt)
        form.addRow("steps/frame", self._steps)
        form.addRow("cell type", self._cell_type)
        form.addRow("field", self._field)

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

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.addLayout(form)
        self._layout.addWidget(self._viewer_btn)
        self._layout.addLayout(buttons)

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

    @staticmethod
    def _spin(value, low, high, step, decimals):
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def params(self):
        """Collect the current control values for the solver driver."""
        return dict(gamma=self._gamma.value(),
                    density=self._density.value(),
                    pressure=self._pressure.value(),
                    mach=self._mach.value(),
                    angle=self._angle.value(),
                    time_increment=self._dt.value(),
                    cell_type=self._cell_type.currentText())

    def field(self):
        return self._field.currentText()

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

    def _on_field_changed(self, name):
        if self.field_changed is not None:
            self.field_changed(name)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
