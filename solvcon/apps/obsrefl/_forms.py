# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the reflection control column, which edit the run
parameters and what the figure shows."""

import math

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLabel,
                               QLineEdit, QSizePolicy, QSpinBox)

from ... import pilot
from ...pilot import panel

from . import _session
from . import _setup
from . import _viewmodel

__all__ = [
    'DisplayGroup',
    'FreeStreamGroup',
    'NumericsGroup',
]


_LABEL_WIDTH = 92
_SETUP = _setup.ReflectionSetup


class _NumberField(QLineEdit):
    """A numeric field that keeps what was typed until it is read, so an
    edit in progress is never rewritten under the cursor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("number")
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(56)

    def set_value(self, value):
        """Show a value with every digit it needs to read back the same,
        so an applied value survives a Reset untouched.

        :param value: The number to show.
        """
        text = f"{value:.12g}"
        if float(text) != value:
            text = repr(float(value))
        self.setText(text)

    def set_invalid(self, invalid):
        """Outline the field as wrong, or clear the outline.

        :param invalid: Whether the value is wrong.
        """
        pilot.set_style_property(self, "invalid",
                                 "true" if invalid else "false")


class _SetupGroup(panel.ColumnGroup):
    """A group of fields of the :class:`~._setup.ReflectionSetup`, which
    wait for Reset.

    Fields are checked as typed: the owner builds the setup from what
    :meth:`read` returns and hands back what the setup rejects, and a
    rejected field is outlined and named under the grid with the range it
    must lie in.  The group is passive: it shows what it is handed and calls
    back into its owner.

    :param title: The title on the head of the group.
    """

    #: The setup fields of the group, in the order of the form.
    KEYS = ()
    #: What the form labels each number field by, and its tip.
    LABELS = {}

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        #: Owner-supplied callback fired on any edit.
        self.edited = None
        self._applied = None
        #: The number fields by setup field name.
        self.fields = {}
        self._grid = self._body_grid(labels=_LABEL_WIDTH)
        self._row = 0
        for key, (text, tip) in self.LABELS.items():
            field = _NumberField()
            field.setToolTip(tip)
            field.textEdited.connect(self._on_edit)
            self.fields[key] = field
            self._add_row(text, field, tip)
        self._grid.setColumnStretch(1, 1)

    def _add_row(self, text, widget, tip=""):
        self._grid.addWidget(panel.form_label(text, tip), self._row, 0)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1

    def _add_messages(self):
        """Put the hint and the error under the rows."""
        self._hint = panel.message("hint")
        self._error = panel.message("fieldError")
        for label in (self._hint, self._error):
            self._grid.addWidget(label, self._row, 0, 1, 2)
            self._row += 1

    def _parse(self, key, text):
        """Return the number a field holds.

        :param key: The setup field the text is for.
        :param text: What the field holds.
        :raise ValueError: Saying what is wrong with the text.
        """
        title = _SETUP.TITLES[key]
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{title} is not a number.") from None
        if not math.isfinite(value):
            raise ValueError(f"{title} must be finite.")
        if key in _SETUP.INTEGERS and value.is_integer():
            return int(value)
        return value

    def show_setup(self, setup):
        """Fill the fields from a setup and take it as the applied one.

        :param setup: The :class:`~._setup.ReflectionSetup` to show.
        """
        self._applied = setup
        for key, field in self.fields.items():
            field.set_value(getattr(setup, key))
        self.show_errors({})

    def read(self):
        """Read the group: ``(values, errors)``.

        ``values`` maps the setup fields that hold a number to it;
        ``errors`` maps the fields that do not to what is wrong with them.
        The ranges a number must lie in belong to the setup, which says so
        when it checks the values.
        """
        values, errors = {}, {}
        for key, field in self.fields.items():
            try:
                values[key] = self._parse(key, field.text())
            except ValueError as error:
                errors[key] = str(error)
        return values, errors

    def show_errors(self, errors):
        """Outline the fields of the group that are wrong and say what is
        wrong with each; the errors of other groups are left to them.

        :param errors: The messages by setup field name, as
            :meth:`~._setup.ReflectionSetup.check` returns them; an empty
            mapping clears the marks.
        """
        for key, field in self.fields.items():
            field.set_invalid(key in errors)
        panel.say(self._error, "\n".join(
            errors[key] for key in self.KEYS if key in errors))

    def error_text(self):
        """Return what the group says is wrong, or an empty string."""
        return "" if self._error.isHidden() else self._error.text()

    def invalid_fields(self):
        """Return the keys of the fields outlined as wrong."""
        return [key for key, field in self.fields.items()
                if "true" == field.property("invalid")]

    def is_dirty(self):
        """Return whether the group differs from the setup last applied."""
        if self._applied is None:
            return False
        values, errors = self.read()
        if errors:
            return True
        return any(not _same(values[key], getattr(self._applied, key))
                   for key in values)

    def set_dirty_hint(self, dirty):
        """Say that edits wait for Reset, or nothing.

        :param dirty: Whether edits wait.
        """
        panel.say(self._hint, "Edited: Reset applies the changes."
                  if dirty else "")

    def _on_edit(self, *_args):
        if self.edited is not None:
            self.edited()


def _same(value, applied):
    if isinstance(value, str) or isinstance(applied, str):
        return value == applied
    return math.isclose(value, applied, rel_tol=1e-9, abs_tol=1e-12)


class FreeStreamGroup(_SetupGroup):
    """The free stream and the incident shock the next Reset builds."""

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


class NumericsGroup(_SetupGroup):
    """The mesh and the time step the next Reset builds, and the step cap
    and the steps per frame, which apply to the standing run at once.
    """

    KEYS = _SETUP.NUMERICS
    LABELS = {
        'nx': ("nx", "Mesh intervals along x; the domain is four units "
                     "long, so nx = 4 ny keeps the cells square"),
        'ny': ("ny", "Mesh intervals along y, over a domain one unit "
                     "tall"),
        'time_increment': ("time step", "The time a step marches; a finer "
                                        "mesh needs a shorter one"),
    }
    #: The highest step cap offered, past any run worth watching.
    STEP_CAP_MAX = 1000000

    def __init__(self, parent=None):
        super().__init__("Numerics", parent)
        #: Owner-supplied callback fired with the step cap and the steps per
        #: frame when either changes.
        self.run_changed = None
        self._cells = QComboBox()
        self._cells.addItems(_SETUP.CELL_TYPES)
        self._cells.setToolTip("The cells the mesh is cut into")
        self._cells.currentIndexChanged.connect(self._on_edit)
        self._add_row("cell type", self._cells, self._cells.toolTip())
        self._add_messages()
        now = QLabel("These apply at once:")
        now.setObjectName("note")
        self._grid.addWidget(now, self._row, 0, 1, 2)
        self._row += 1
        self._cap = QSpinBox()
        self._cap.setRange(1, self.STEP_CAP_MAX)
        self._cap.setToolTip("The step the run stops at")
        self._frame = QSpinBox()
        self._frame.setRange(1, 1000)
        self._frame.setToolTip("Steps marched between two drawn frames")
        self._add_row("step cap", self._cap, self._cap.toolTip())
        self._add_row("steps / frame", self._frame, self._frame.toolTip())
        self.set_run_settings(_session.ReflectionSession.MAX_STEPS,
                              _session.ReflectionSession.STEPS_PER_CHUNK)
        panel.guard_wheel(self._cells, self._cap, self._frame)
        for spin in (self._cap, self._frame):
            # Applied as typed, the "5" of a new cap of 5000 would end the
            # run at once.
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._on_run)

    def show_setup(self, setup):
        """Fill the fields and the cell type from a setup and take it as
        the applied one.

        :param setup: The :class:`~._setup.ReflectionSetup` to show.
        """
        with QSignalBlocker(self._cells):
            self._cells.setCurrentText(setup.cell_type)
        super().show_setup(setup)

    def read(self):
        """Read the group as :meth:`_SetupGroup.read` does, with the cell
        type, which always holds one of the types."""
        values, errors = super().read()
        values['cell_type'] = self._cells.currentText()
        return values, errors

    def run_settings(self):
        """Return the step cap and the steps per frame."""
        return self._cap.value(), self._frame.value()

    def set_run_settings(self, max_steps, steps_per_frame):
        """Show the step cap and the steps per frame without firing the
        callback.

        :param max_steps: The step cap of the run.
        :param steps_per_frame: The steps marched between two frames.
        """
        for spin, value in ((self._cap, max_steps),
                            (self._frame, steps_per_frame)):
            with QSignalBlocker(spin):
                spin.setValue(value)

    def _on_run(self, _value):
        if self.run_changed is not None:
            self.run_changed(*self.run_settings())


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
        grid = self._body_grid(labels=_LABEL_WIDTH)
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
