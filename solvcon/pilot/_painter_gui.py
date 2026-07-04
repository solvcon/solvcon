# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Painter toolbox for the 2D canvas.
"""

from PySide6 import QtCore, QtWidgets

from . import _gui_common
from ._pilot_core import draw_tool_names, default_draw_tool_name

__all__ = [
    'Painter',
]


class Painter(_gui_common.PilotFeature):
    """
    Tool palette for drawing shapes on a 2D canvas, toggled from the View
    "Panels" submenu. The selected tool is held by the manager and applied
    to the focused 2D canvas, not bound to any one canvas.
    """

    # Button label for a tool id. The ids, their order, and the default
    # come from the C++ registry; this only supplies human-facing text.
    # A tool with no entry here falls back to its title-cased id.
    TOOL_LABELS = {
        "pan": "Pan / Move",
        "line": "Line",
        "triangle": "Triangle",
        "rectangle": "Rectangle",
        "ellipse": "Ellipse",
        "circle": "Circle",
    }

    def __init__(self, *args, **kw):
        super(Painter, self).__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._buttons = {}

    def populate_menu(self):
        """Add a checkable "Painter" toggle under the View/Panels path."""
        self._action = self.add_action(
            "View/Panels", "Painter", "Toggle the Painter toolbox", None,
            id="panel.painter", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        """Show or hide the toolbox dock from the menu toggle."""
        if checked:
            self._ensure_dock()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_dock(self):
        """Create the dock and its tool buttons once, lazily."""
        if self._dock is not None:
            return
        dock = QtWidgets.QDockWidget("Painter", self._mainWindow)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        body = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(body)
        group = QtWidgets.QButtonGroup(body)
        group.setExclusive(True)

        for tool in draw_tool_names():
            button = QtWidgets.QToolButton(body)
            button.setText(self.TOOL_LABELS.get(tool, tool.title()))
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, t=tool: self._select_tool(t))
            group.addButton(button)
            layout.addWidget(button)
            self._buttons[tool] = button

        layout.addStretch(1)
        dock.setWidget(body)
        self._mainWindow.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        # Keep the menu check in sync when the dock is closed by its button.
        if self._action is not None:
            dock.visibilityChanged.connect(self._action.setChecked)
        self._dock = dock

    def present(self):
        """
        Show the toolbox dock and reset the focused canvas to the Pan tool.
        """
        self._ensure_dock()
        default_tool = default_draw_tool_name()
        self._buttons[default_tool].setChecked(True)
        self._select_tool(default_tool)
        self._dock.show()
        self._dock.raise_()

    def _select_tool(self, tool):
        """Set the selected tool; the manager applies it to the focused
        canvas and re-applies it as focus moves between canvases."""
        self._mgr.setDrawTool(tool)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
