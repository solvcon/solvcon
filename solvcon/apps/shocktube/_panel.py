# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The configuration column of the shocktube app.

The configuration uses a panel docked beside the app window.  It contains
foldable sections to configure the app, including problem, the grid, and the
march.
"""

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QFrame, QGridLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

__all__ = [
    'ConfigColumn',
    'FoldGroup',
]


ARROW_OPEN = chr(0x25be)
ARROW_FOLDED = chr(0x25b8)

#: What a message names each config value by.
TITLES = {'gamma': "Gamma", 'p_left': "Left pressure",
          'rho_left': "Left density", 'p_right': "Right pressure",
          'rho_right': "Right density", 'xmin': "x min", 'xmax': "x max",
          'ncoord': "Points", 'time_increment': "The time step",
          'max_steps': "Steps", 'time_interval': "The interval"}
#: The values the solver takes as whole numbers.
WHOLE = ('ncoord', 'max_steps', 'time_interval')
#: The outline of a field the column rejects, which reads in either theme.
INVALID_SHEET = "QLineEdit { border: 1px solid #d0453a; }"


def check(values):
    """Return what is wrong with ``values`` of the config by name, an
    empty mapping when the solver can take them."""
    errors = {}
    for key in ('p_left', 'rho_left', 'p_right', 'rho_right',
                'time_increment'):
        if values[key] <= 0:
            errors[key] = f"{TITLES[key]} must be positive."
    if values['gamma'] <= 1:
        errors['gamma'] = "Gamma must be greater than 1."
    if values['xmax'] <= values['xmin']:
        errors['xmax'] = "x max must be greater than x min."
    # The solver takes an odd count, and keeps every other point.
    if values['ncoord'] < 5 or 0 == values['ncoord'] % 2:
        errors['ncoord'] = "Points must be odd and at least 5."
    for key in ('max_steps', 'time_interval'):
        if values[key] < 1:
            errors[key] = f"{TITLES[key]} must be at least 1."
    return errors


class FoldGroup(QWidget):
    """A group of the column that folds behind its head.  The head carries
    a summary on its right, so a folded group still says what it holds.

    :ivar is_open: Whether the body shows.
    :ivar toggled: Owner-supplied callback fired with the group when its
        head is clicked.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_open = True
        self.toggled = None
        self._head = QPushButton()
        self._head.setFlat(True)
        self._head.setStyleSheet("text-align: left; padding: 4px 2px;")
        self._head.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._head.setCursor(Qt.PointingHandCursor)
        font = QFont(self.font())
        font.setWeight(QFont.DemiBold)
        self._head.setFont(font)
        self._head.clicked.connect(self._on_head)
        self._summary = QLabel("")
        # The summary lets the mouse through to the head under it.
        self._summary.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._summary.setEnabled(False)
        self.body = QWidget()
        head = QGridLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self._head, 0, 0)
        head.addWidget(self._summary, 0, 0, Qt.AlignRight | Qt.AlignVCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.body)
        self._sync()

    def _sync(self):
        arrow = ARROW_OPEN if self.is_open else ARROW_FOLDED
        self._head.setText(f"{arrow}  {self.title.upper()}")
        self.body.setVisible(self.is_open)

    def set_open(self, open_):
        """Show the body, or fold it behind the head."""
        self.is_open = bool(open_)
        self._sync()

    def summary(self):
        """Return the text on the right of the head."""
        return self._summary.text()

    def set_summary(self, text):
        """Say ``text`` on the right of the head."""
        self._summary.setText(text)
        self._head.setToolTip(text)

    def _on_head(self):
        if self.toggled is not None:
            self.toggled(self)
        else:
            self.set_open(not self.is_open)


class ConfigColumn(QScrollArea):
    """The config column: a group each for the problem, the grid, and the
    march, over the button that applies them.

    Fields are checked as typed.  A field the solver cannot take is
    outlined and named under the groups, and Apply waits until none is
    left.  One group is open at a time, so the column stays short.

    :ivar fields: The field of each config value by name.
    :ivar groups: The :class:`FoldGroup` of the problem, the grid, and the
        march.
    :ivar applied: Owner-supplied callback fired with the values by name
        when Apply is clicked.
    """

    #: The title and the rows of each group: the label and the config
    #: names it heads, one per column.
    GROUPS = (
        ("Problem", (("pressure", ('p_left', 'p_right')),
                     ("density", ('rho_left', 'rho_right')),
                     ("gamma", ('gamma',)))),
        ("Grid", (("x min", ('xmin',)), ("x max", ('xmax',)),
                  ("points", ('ncoord',)))),
        ("March", (("time step", ('time_increment',)),
                   ("steps", ('max_steps',)),
                   ("interval (ms)", ('time_interval',)))),
    )
    WIDTH = 280

    def __init__(self, parent=None):
        super().__init__(parent)
        self.applied = None
        self.fields = {}
        self.groups = []
        self._last = {}
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(2)
        for title, rows in self.GROUPS:
            group = self._build_group(title, rows)
            group.toggled = self._on_group
            group.set_open(not self.groups)
            self.groups.append(group)
            layout.addWidget(group)
        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.hide()
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setEnabled(False)
        self._hint.hide()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip(
            "Build the problem anew from the fields and start over")
        self.apply_button.clicked.connect(self._on_apply)
        layout.addSpacing(6)
        layout.addWidget(self._error)
        layout.addWidget(self._hint)
        layout.addWidget(self.apply_button)
        layout.addStretch(1)
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedWidth(self.WIDTH)

    def _build_group(self, title, rows):
        group = FoldGroup(title)
        grid = QGridLayout(group.body)
        grid.setContentsMargins(6, 2, 4, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        top = 0
        if any(len(keys) > 1 for _, keys in rows):
            for col, text in ((1, "left"), (2, "right")):
                head = QLabel(text)
                head.setAlignment(Qt.AlignRight)
                head.setEnabled(False)
                grid.addWidget(head, 0, col)
            top = 1
        for row, (text, keys) in enumerate(rows, top):
            grid.addWidget(QLabel(text), row, 0)
            for col, key in enumerate(keys, 1):
                field = QLineEdit()
                field.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                field.setMinimumWidth(56)
                field.setToolTip(TITLES[key])
                field.textEdited.connect(self._on_edit)
                self.fields[key] = field
                grid.addWidget(field, row, col)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return group

    def show_values(self, values):
        """Fill the fields from ``values`` by name and take them as the
        applied ones."""
        self._last = dict(values)
        for key, value in values.items():
            # Every digit it needs to read back the same, so an untouched
            # field never reads as an edit.
            text = f"{value:.12g}"
            self.fields[key].setText(
                text if float(text) == value else repr(float(value)))
        self._on_edit()

    def read(self):
        """Read the fields: ``(values, errors)``, ``values`` the numbers by
        name and ``errors`` what is wrong with the fields by name."""
        values, errors = {}, {}
        for key, field in self.fields.items():
            try:
                value = float(field.text())
            except ValueError:
                errors[key] = f"{TITLES[key]} is not a number."
                continue
            if not math.isfinite(value):
                errors[key] = f"{TITLES[key]} must be finite."
            elif key in WHOLE and not value.is_integer():
                errors[key] = f"{TITLES[key]} must be a whole number."
            else:
                values[key] = int(value) if key in WHOLE else value
        if not errors:
            errors = check(values)
        return values, errors

    def error_text(self):
        """Return what the column says is wrong, or an empty string."""
        return "" if self._error.isHidden() else self._error.text()

    def invalid_fields(self):
        """Return the names of the fields outlined as wrong."""
        return [key for key, field in self.fields.items()
                if field.styleSheet()]

    def is_dirty(self):
        """Return whether the fields differ from the values last
        applied."""
        values, errors = self.read()
        return bool(errors) or values != self._last

    def _on_group(self, group):
        """Open the clicked group and fold the others, or fold it."""
        opening = not group.is_open
        for other in self.groups:
            other.set_open(opening and other is group)

    def _on_edit(self, _text=None):
        values, errors = self.read()
        for key, field in self.fields.items():
            field.setStyleSheet(INVALID_SHEET if key in errors else "")
        self._error.setText("\n".join(errors.values()))
        self._error.setVisible(bool(errors))
        pending = not errors and values != self._last
        self._hint.setText("Edited: Apply starts over with the changes."
                           if pending else "")
        self._hint.setVisible(pending)
        self.apply_button.setEnabled(not errors)
        shown = self._last if errors else values
        if shown:
            self._summarize(shown)

    def _summarize(self, values):
        problem, grid, march = self.groups
        problem.set_summary(
            "p {p_left:g} | {p_right:g}, rho {rho_left:g} | {rho_right:g}"
            .format(**values))
        grid.set_summary("{ncoord} on [{xmin:g}, {xmax:g}]".format(**values))
        march.set_summary("{max_steps} x {time_increment:g}".format(**values))

    def _on_apply(self):
        values, errors = self.read()
        if errors:
            return
        self._last = dict(values)
        self._on_edit()
        if self.applied is not None:
            self.applied(values)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
