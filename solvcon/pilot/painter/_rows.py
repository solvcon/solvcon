# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The object row the Painter inspector lists.

The design draws the same row twice, filling the Layers page and the short
list the Design page still stands in for, so the widget lives here rather
than in either page.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from ._style import Parts

__all__ = [
    'ICON_PX',
    'HEIGHT',
    'ObjectRow',
]

#: The type icon's size and the row's own, in device-independent pixels.
ICON_PX = 13
HEIGHT = 30

_MARGINS = (8, 0, 8, 0)
_GAP = 8


class ObjectRow(QtWidgets.QFrame):
    """One object row: a type icon, the object's name, and its key metric.

    The row wears the accent tint while the canvas has its object selected.
    That state is a Qt property rather than a second style sheet, so a page
    writes one sheet for every row and Qt re-reads it as the selection moves.

    A press hands the shape it stands for to whoever owns the list; the row
    neither reaches the canvas nor marks itself, so the selection it shows
    stays the one the canvas reports.
    """

    picked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("row")
        self.setFixedHeight(HEIGHT)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self._shape_id = -1
        self._icon = QtWidgets.QLabel(self)
        self._icon.setFixedWidth(ICON_PX)
        self._name = QtWidgets.QLabel(self)
        self._name.setObjectName("name")
        self._metric = QtWidgets.QLabel(self)
        self._metric.setObjectName("metric")
        self._metric.setFont(Parts.mono_font())
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*_MARGINS)
        layout.setSpacing(_GAP)
        layout.addWidget(self._icon)
        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._metric)

    @property
    def name(self):
        """The object name the row shows."""
        return self._name.text()

    @property
    def metric(self):
        """The measurement the row shows."""
        return self._metric.text()

    @property
    def icon(self):
        """The type icon the row shows."""
        return self._icon.pixmap()

    @property
    def metric_color(self):
        """The color the metric reads in, as the style sheet resolved it."""
        return self._metric.palette().color(QtGui.QPalette.WindowText)

    def selected(self):
        """Whether the row stands for the canvas's selection."""
        return bool(self.property("selected"))

    def show_object(self, shape_id, name, metric, icon, selected):
        """Show the shape ``shape_id``, ``icon`` being its pixmap or ``None``
        for a type the icon set does not draw."""
        self._shape_id = shape_id
        self._name.setText(name)
        self._metric.setText(metric)
        if icon is None:
            self._icon.clear()
        else:
            self._icon.setPixmap(icon)
        if selected != self.selected():
            self.setProperty("selected", selected)
            # A property the style sheet selects on only reaches the paint
            # after the widget is polished against the sheet again. The metric
            # is colored by a rule that reads the property off the row, and Qt
            # resolves a widget's rules against its own cache, so the label
            # has to be polished alongside the row it hangs from.
            for widget in (self, self._metric):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if QtCore.Qt.LeftButton == event.button():
            self.picked.emit(self._shape_id)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
