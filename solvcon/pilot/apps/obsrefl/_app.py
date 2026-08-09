# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Pilot feature that runs the 2D Euler oblique-shock reflection and draws a
selected solution field as a flat color map.

The feature is the composition root of the app: a toggle in the View
"Panels" submenu owns the dock, and building the dock builds the control
panel from :mod:`._panel`, the domain viewer from :mod:`._viewer`, and the
:class:`~._control.RunController` that wires the two around the run
session.  The feature itself holds no run logic; it only composes the
parts and forwards the inspector nudge to the outer GUI.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget

from ...base import _gui_common
from ._control import RunController
from ._panel import SolutionPanel
from ._viewer import DomainViewer

__all__ = [  # noqa: F822
    'ObliqueShockApp',
]


class ObliqueShockApp(_gui_common.PilotFeature):
    """Euler solver panel, toggled from the View "Panels" submenu.

    The panel owns one domain viewer sub-window and one solver run; the
    controller between them starts, pauses, steps, and stops the march.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Fired after a run sets the viewer mesh, so the inspector can refresh.
        self.viewer_updated = None
        self._action = None
        self._dock = None
        self._panel = None
        self._control = None
        self._viewer = DomainViewer(self._mgr)
        self._viewer.mesh_updated = self._notify_viewer_updated

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Euler solver", "Toggle the Euler solver panel",
            None, id="panel.euler_solver", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        if checked:
            self._ensure_panel()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        if self._panel is not None:
            return
        self._panel = SolutionPanel()
        self._control = RunController(self._panel, self._viewer)
        self._dock = QDockWidget("euler solver")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)
        self._control.preview()

    def _notify_viewer_updated(self):
        if self.viewer_updated is not None:
            self.viewer_updated()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
