# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Painter toolbox for the 2D canvas.
"""

from PySide6 import QtCore, QtWidgets

from .. import _gui_common
from .._pilot_core import draw_tool_names, default_draw_tool_name

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
        self._tool_group = None
        self._tool_actions = {}

    def populate_menu(self):
        """Add the Painter toggle and the draw-tool radio group."""
        self._action = self.add_action(
            "View/Panels", "Painter", "Toggle the Painter toolbox", None,
            id="panel.painter", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)
        self._build_tool_actions()

    def _build_tool_actions(self):
        """One exclusive checkable action per draw tool, the single source of
        truth shared by the Canvas/Draw tool radio items and the toolbox
        buttons. Each action routes its own trigger to the manager.

        Idempotent: the tools are declared once on the model, so a second
        Painter reuses the existing actions instead of duplicating them.
        """
        if self._tool_actions:
            return
        model = self._mgr.menu_model
        model.menu("Canvas/Draw tool", weight=10)
        # Held by the model under a group id so the selection is queryable.
        self._tool_group = model.group("draw.tool")
        self._tool_group.setExclusive(True)
        mgr = self._mgr
        weight = 10
        created = False
        for tool in draw_tool_names():
            act = model.action("draw.tool." + tool)
            if act is None:
                label = self.TOOL_LABELS.get(tool, tool.title())
                act = self.add_action(
                    "Canvas/Draw tool", label, f"Draw with the {label} tool",
                    lambda t=tool: mgr.setDrawTool(t),
                    id="draw.tool." + tool, weight=weight, checkable=True)
                self._tool_group.addAction(act)
                created = True
            self._tool_actions[tool] = act
            weight += 10
        if created:
            self._tool_actions[default_draw_tool_name()].setChecked(True)

    def _on_toggled(self, checked):
        """Show or hide the toolbox dock from the menu toggle."""
        if checked:
            self._ensure_dock()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_dock(self):
        """Create the dock once, its buttons views of the tool actions."""
        if self._dock is not None:
            return
        # A standalone Painter (used in tests) reaches the dock without
        # populate_menu, so make sure the tool actions exist first.
        self._build_tool_actions()
        dock = QtWidgets.QDockWidget("Painter", self._mainWindow)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        body = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(body)
        for tool in draw_tool_names():
            button = QtWidgets.QToolButton(body)
            # The default action drives the button and reflects its checked
            # state, so a menu radio and a button stay one selection.
            button.setDefaultAction(self._tool_actions[tool])
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
        """Show the toolbox dock and reset the focused canvas to the default
        tool; the action group updates every surface."""
        self._ensure_dock()
        self._mgr.setDrawTool(default_draw_tool_name())
        self._dock.show()
        self._dock.raise_()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
