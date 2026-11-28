# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The "App config" dock panel, which holds the config of the active app.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QStackedWidget

from .. import base
from . import _chrome

__all__ = [
    'AppConfig',
]


class AppConfig(base.PilotFeature):
    """One dock panel for the config of every app.

    An app adds its config widget with the sub-window it configures, and
    the panel shows the widget of the active one.  The panel lives while
    an app sub-window is open, and its toggle in "View > Panels" is
    enabled only then.

    :ivar dock: The dock panel, or None while no app is open.
    :ivar action: The "View > Panels" item that toggles :attr:`dock`.
    """

    #: The title of the dock panel and of its toggle.
    TITLE = "App config"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.dock = None
        self.action = None
        self._stack = None
        self._entries = []

    def populate_menu(self):
        self.action = self.add_action(
            "View/Panels", self.TITLE,
            "Toggle the config panel of the active app", None,
            id="panel.app_config", weight=16, checkable=True)
        self.action.setEnabled(False)
        self.action.toggled.connect(self._on_toggled)
        self._mgr.mdiArea.subWindowActivated.connect(self.show_for)

    def add(self, subwindow, widget):
        """Hold ``widget`` as the config of ``subwindow`` and show it."""
        if self.dock is None:
            self._stack = QStackedWidget()
            self.dock = QDockWidget(self.TITLE)
            self.dock.setWidget(self._stack)
            self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                               self.dock)
            self.dock.visibilityChanged.connect(self._sync_action)
            self.action.setEnabled(True)
        self._entries.append((subwindow, widget))
        self._stack.addWidget(widget)
        self._stack.setCurrentWidget(widget)
        self._note(subwindow)
        subwindow.windowTitleChanged.connect(
            lambda _title, held=subwindow: self._note(held))
        self.dock.show()
        self._sync_action()

    def remove(self, subwindow):
        """Drop the config of ``subwindow``, and the panel with the last."""
        widgets = [w for s, w in self._entries if s is subwindow]
        self._entries = [(s, w) for s, w in self._entries
                         if s is not subwindow]
        for widget in widgets:
            self._stack.removeWidget(widget)
            widget.deleteLater()
        for held, _ in self._entries:
            self._note(held)
        if self.dock is not None and not self._entries:
            self._mgr.mainWindow.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
            self._stack = None
            self.action.setChecked(False)
            self.action.setEnabled(False)

    def show_for(self, subwindow):
        """Show the config of ``subwindow`` if it has one."""
        for held, widget in self._entries:
            if held is subwindow:
                self._stack.setCurrentWidget(widget)
                self._note(subwindow)

    def _note(self, subwindow):
        # The panel names the window it follows, which is not always the
        # one in front. An app window is titled "<app>: <run>".
        shown = (self._stack.currentWidget() if self._stack is not None
                 else None)
        if any(s is subwindow and w is shown for s, w in self._entries):
            app, _, _ = subwindow.windowTitle().partition(":")
            _chrome.set_dock_note(self.dock, app)

    def _on_toggled(self, checked):
        if self.dock is not None:
            self.dock.setVisible(checked)

    def _sync_action(self, _visible=None):
        # Read the dock rather than the signal, which also says False for
        # a dock tabbed behind another.
        self.action.setChecked(not self.dock.isHidden())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
