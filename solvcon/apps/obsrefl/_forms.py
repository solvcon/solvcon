# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the reflection control column, which edit the run
parameters."""

import math

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QLineEdit, QSizePolicy

from ... import pilot

from . import _setup
from . import _widgets

__all__ = [
    'FreeStreamGroup',
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
        with QSignalBlocker(self):
            self.setText(text)

    def set_invalid(self, invalid):
        """Outline the field as wrong, or clear the outline.

        :param invalid: Whether the value is wrong.
        """
        pilot.set_style_property(self, "invalid",
                                 "true" if invalid else "false")


class _SetupGroup(_widgets.FoldGroup):
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
        self._grid.addWidget(_widgets.form_label(text, tip), self._row, 0)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1

    def _add_messages(self):
        """Put the hint and the error under the rows."""
        self._hint = _widgets.message("hint")
        self._error = _widgets.message("fieldError")
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
        _widgets.say(self._error, "\n".join(
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
        _widgets.say(self._hint, "Edited: Reset applies the changes."
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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
