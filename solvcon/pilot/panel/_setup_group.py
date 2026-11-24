# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of number fields a control column edits the setup of an app
run with."""

import math

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QLabel, QLineEdit, QSizePolicy, QSpinBox

from .. import _style
from . import _app_widgets

__all__ = [
    'NumberField',
    'SetupGroup',
]


class NumberField(QLineEdit):
    """A numeric field that holds what is typed until the owner reads it,
    and shows a value only through :meth:`set_value`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("number")
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(56)

    def set_value(self, value):
        """Show a value with every digit it needs to read back the same.

        :param value: The number to show.
        """
        # A value that reads back the same leaves the group clean after a
        # Reset.
        text = f"{value:.12g}"
        if float(text) != value:
            text = repr(float(value))
        self.setText(text)

    def set_invalid(self, invalid):
        """Outline the field as wrong, or clear the outline.

        :param invalid: Whether the value is wrong.
        """
        _style.set_style_property(self, "invalid",
                                  "true" if invalid else "false")


class SetupGroup(_app_widgets.ColumnGroup):
    """A group of number fields of the setup of an app run, whose edits
    wait for Reset.

    The owner sets the :attr:`edited` callback, builds the setup from what
    :meth:`read` returns, and hands what the setup rejects to
    :meth:`show_errors`, which outlines each such field and says what is
    wrong under the grid.

    A subclass names the setup class in :attr:`SETUP`, which carries
    ``TITLES``, what a message calls each field, and ``INTEGERS``, the
    fields that take whole numbers, and the fields in :attr:`KEYS` and
    :attr:`LABELS`.  It puts the messages under its rows with
    :meth:`_add_messages`, and may add the run settings with
    :meth:`_add_run_settings`.

    :param title: The title on the head of the group.
    """

    #: The setup class of the app, a frozen dataclass.
    SETUP = None
    #: The setup fields of the group, in the order of the form.
    KEYS = ()
    #: What the form labels each number field by, and its tip.
    LABELS = {}
    #: The least width of the column of labels.
    LABEL_WIDTH = 92
    #: The highest step cap :meth:`_add_run_settings` offers.
    STEP_CAP_MAX = 1000000

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        #: Owner-supplied callback fired on any edit.
        self.edited = None
        #: Owner-supplied callback fired with the step cap and the steps per
        #: frame when either changes, in a group with
        #: :meth:`_add_run_settings`.
        self.run_changed = None
        self._applied = None
        #: The number fields by setup field name.
        self.fields = {}
        self._grid = self._body_grid(labels=self.LABEL_WIDTH)
        self._row = 0
        for key, (text, tip) in self.LABELS.items():
            field = NumberField()
            field.setToolTip(tip)
            field.textEdited.connect(self._on_edit)
            self.fields[key] = field
            self._add_row(text, field, tip)
        self._grid.setColumnStretch(1, 1)

    def _add_row(self, text, widget, tip=""):
        self._grid.addWidget(_app_widgets.form_label(text, tip), self._row, 0)
        self._grid.addWidget(widget, self._row, 1)
        self._row += 1

    def _add_messages(self):
        """Put the hint and the error under the rows."""
        self._hint = _app_widgets.message("hint")
        self._error = _app_widgets.message("fieldError")
        for label in (self._hint, self._error):
            self._grid.addWidget(label, self._row, 0, 1, 2)
            self._row += 1

    def _add_run_settings(self, max_steps, steps_per_frame):
        """Put the step cap and the steps per frame under the rows, which
        apply to the standing run at once rather than wait for Reset.

        :param max_steps: The step cap to start with.
        :param steps_per_frame: The steps per frame to start with.
        """
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
        self.set_run_settings(max_steps, steps_per_frame)
        _app_widgets.guard_wheel(self._cap, self._frame)
        for spin in (self._cap, self._frame):
            # Applied as typed, the "5" of a new cap of 5000 would end the
            # run at once.
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._on_run)

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

    def _parse(self, key, text):
        """Return the number a field holds.

        :param key: The setup field the text is for.
        :param text: What the field holds.
        :raise ValueError: Saying what is wrong with the text.
        """
        title = self.SETUP.TITLES[key]
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{title} is not a number.") from None
        if not math.isfinite(value):
            raise ValueError(f"{title} must be finite.")
        if key in self.SETUP.INTEGERS and value.is_integer():
            return int(value)
        return value

    def show_setup(self, setup):
        """Fill the fields from a setup and take it as the applied one.

        :param setup: The setup to show, of :attr:`SETUP`.
        """
        self._applied = setup
        for key, field in self.fields.items():
            field.set_value(getattr(setup, key))
        self.show_errors({})

    def read(self):
        """Read the group: ``(values, errors)``.

        ``values`` maps the setup fields that hold a number to it;
        ``errors`` maps the fields that do not to what is wrong with them.
        The setup checks the ranges.
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

        :param errors: The messages by setup field name, as the ``check()``
            of the setup returns them; an empty mapping clears the marks.
        """
        for key, field in self.fields.items():
            field.set_invalid(key in errors)
        _app_widgets.say(self._error, "\n".join(
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
        _app_widgets.say(self._hint, "Edited: Reset applies the changes."
                         if dirty else "")

    def _on_edit(self, *_args):
        if self.edited is not None:
            self.edited()


def _same(value, applied):
    # A number read back from its text differs from the applied one by
    # rounding at most.
    if isinstance(value, str) or isinstance(applied, str):
        return value == applied
    return math.isclose(value, applied, rel_tol=1e-9, abs_tol=1e-12)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
