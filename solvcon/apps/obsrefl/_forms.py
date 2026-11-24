# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the reflection control column, which edit the run
parameters and what the figure shows."""

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox

from ...pilot import panel

from . import _session
from . import _setup
from . import _viewmodel

__all__ = [
    'DisplayGroup',
    'FreeStreamGroup',
    'NumericsGroup',
]


_SETUP = _setup.ReflectionSetup


class FreeStreamGroup(panel.SetupGroup):
    """The free stream and the incident shock the next Reset builds."""

    SETUP = _SETUP
    KEYS = _SETUP.FREE_STREAM
    LABELS = {
        'gamma': ("gamma", "Ratio of the specific heats"),
        'density': ("density", "Density of the free stream"),
        'pressure': ("pressure", "Pressure of the free stream"),
        'mach': ("Mach", "Mach number of the free stream"),
        'angle': ("angle (deg)", "Deflection of the flow across the "
                                 "incident shock, in degrees"),
    }

    def __init__(self, parent=None):
        super().__init__("Free stream", parent)
        self._add_messages()


class NumericsGroup(panel.SetupGroup):
    """The mesh and the time step the next Reset builds, and the step cap
    and the steps per frame, which apply to the standing run at once.
    """

    SETUP = _SETUP
    KEYS = _SETUP.NUMERICS
    LABELS = {
        'nx': ("nx", "Mesh intervals along x; the domain is four units "
                     "long, so nx = 4 ny keeps the cells square"),
        'ny': ("ny", "Mesh intervals along y, over a domain one unit "
                     "tall"),
        'time_increment': ("time step", "The time a step marches; a finer "
                                        "mesh needs a shorter one"),
    }

    def __init__(self, parent=None):
        super().__init__("Numerics", parent)
        self._cells = QComboBox()
        self._cells.addItems(_SETUP.CELL_TYPES)
        self._cells.setToolTip("The cells the mesh is cut into")
        self._cells.currentIndexChanged.connect(self._on_edit)
        self._add_row("cell type", self._cells, self._cells.toolTip())
        panel.guard_wheel(self._cells)
        self._add_messages()
        self._add_run_settings(_session.ReflectionSession.MAX_STEPS,
                               _session.ReflectionSession.STEPS_PER_CHUNK)

    def show_setup(self, setup):
        """Fill the fields and the cell type from a setup and take it as
        the applied one.

        :param setup: The :class:`~._setup.ReflectionSetup` to show.
        """
        with QSignalBlocker(self._cells):
            self._cells.setCurrentText(setup.cell_type)
        super().show_setup(setup)

    def read(self):
        """Read the group as :meth:`~solvcon.pilot.panel.SetupGroup.read`
        does, with the cell type, which always holds one of the types."""
        values, errors = super().read()
        values['cell_type'] = self._cells.currentText()
        return values, errors


class DisplayGroup(panel.ColumnGroup):
    """What the figure shows of the run: the field, the color bar, the
    marks of the profile cut, and its height.  None of it marches the run,
    so an edit applies at once.  The group is passive: it shows what it is
    handed and calls back into its owner.
    """

    def __init__(self, parent=None):
        super().__init__("Display", parent)
        #: Owner-supplied callback fired with the
        #: :class:`~._viewmodel.Display` on any change.
        self.display_changed = None
        grid = self._body_grid(labels=panel.SetupGroup.LABEL_WIDTH)
        self._field = QComboBox()
        for name, label in _viewmodel.FIELD_LABELS.items():
            self._field.addItem(label, name)
        self._field.setToolTip("The field the cells are colored by")
        self._bar = QCheckBox("color bar")
        self._bar.setToolTip("The scale of the colors, right of the domain")
        self._cells = QCheckBox("mark cells")
        self._cells.setToolTip("Outline the cells the profile samples")
        self._line = QCheckBox("mark line")
        self._line.setToolTip("Draw the line the profile runs along")
        self._cut = QDoubleSpinBox()
        self._cut.setRange(0.0, 1.0)
        self._cut.setSingleStep(0.05)
        self._cut.setDecimals(2)
        self._cut.setToolTip("The height of the profile, a fraction of the "
                             "domain height; the default crosses both "
                             "shocks")
        grid.addWidget(panel.form_label("field", self._field.toolTip()),
                       0, 0)
        grid.addWidget(self._field, 0, 1, 1, 2)
        grid.addWidget(self._bar, 1, 0, 1, 2)
        grid.addWidget(self._cells, 2, 0)
        grid.addWidget(self._line, 2, 1, 1, 2)
        grid.addWidget(panel.form_label("cut height", self._cut.toolTip()),
                       3, 0)
        grid.addWidget(self._cut, 3, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        self._controls = (self._field, self._bar, self._cells, self._line,
                          self._cut)
        panel.guard_wheel(self._field, self._cut)
        self.show_display(_viewmodel.Display())
        self._field.currentIndexChanged.connect(self._on_change)
        for box in (self._bar, self._cells, self._line):
            box.toggled.connect(self._on_change)
        self._cut.valueChanged.connect(self._on_change)

    def display(self):
        """Return the :class:`~._viewmodel.Display` the controls hold."""
        return _viewmodel.Display(
            field=self._field.currentData(), bar=self._bar.isChecked(),
            mark_cells=self._cells.isChecked(),
            mark_line=self._line.isChecked(), cut=self._cut.value())

    def show_display(self, display):
        """Set the controls from a display without firing the callback.

        :param display: The :class:`~._viewmodel.Display` to show.
        """
        blockers = [QSignalBlocker(widget) for widget in self._controls]
        self._field.setCurrentIndex(self._field.findData(display.field))
        self._bar.setChecked(display.bar)
        self._cells.setChecked(display.mark_cells)
        self._line.setChecked(display.mark_line)
        self._cut.setValue(display.cut)
        for blocker in blockers:
            blocker.unblock()

    def _on_change(self, *_args):
        if self.display_changed is not None:
            self.display_changed(self.display())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
