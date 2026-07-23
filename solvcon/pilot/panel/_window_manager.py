# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Window manager feature for pilot.

List every open MDI sub-window under the "Window" menu and bring one to
the foreground when its entry is chosen.
"""

from PySide6 import QtGui

from ..base import _gui_common

__all__ = [
    'WindowManager',
]


class WindowManager(_gui_common.PilotFeature):
    """List open MDI sub-windows under the "Window" menu.

    One checkable action per open sub-window, labelled by its title.
    Triggering an action activates that sub-window; the active one is
    checked. The list is rebuilt each time the menu is about to show.
    """

    #: objectName tagging every dynamic per-sub-window action.
    ITEM_ID = "window.subwindow"

    def __init__(self, *args, **kw):
        super(WindowManager, self).__init__(*args, **kw)
        self._menu = None
        self._items = []

    def populate_menu(self):
        """Anchor the dynamic list on the "Window" menu.

        The menu holds nothing static (panel toggles live under
        View/Panels), so the list is seeded right away: a native menu
        bar hides an empty menu, and a hidden menu can never fire
        aboutToShow to fill itself.
        """
        self._menu = self._mgr.menu_model.menu("Window")
        self._menu.aboutToShow.connect(self._rebuild)
        self._rebuild()

    def _rebuild(self):
        """Refresh the sub-window list to match the MDI area.

        Drop the actions from the previous show, then append one checkable
        action per visible sub-window in area order, checking the active
        one. A disabled placeholder is shown when none are open.
        """
        for act in self._items:
            self._menu.removeAction(act)
            act.deleteLater()
        self._items = []

        mdi = self._mgr.mdiArea
        active = mdi.activeSubWindow()
        subwins = [s for s in mdi.subWindowList() if s.isVisible()]

        if not subwins:
            self._append_placeholder()
            return

        for index, subwin in enumerate(subwins):
            self._append_item(index, subwin, subwin is active)

    def _append_item(self, index, subwin, is_active):
        """Append one checkable action that activates ``subwin``."""
        title = subwin.windowTitle() or "window"
        act = QtGui.QAction("%s" % (title), self._menu)
        act.setObjectName(self.ITEM_ID)
        act.setStatusTip("Bring '%s' to the foreground" % title)
        act.setCheckable(True)
        act.setChecked(is_active)
        act.triggered.connect(
            lambda checked=False, s=subwin: self._activate(s))
        self._menu.addAction(act)
        self._items.append(act)

    def _append_placeholder(self):
        """Append a disabled hint when no sub-window is open."""
        act = QtGui.QAction("(No open windows)", self._menu)
        act.setEnabled(False)
        self._menu.addAction(act)
        self._items.append(act)

    def _activate(self, subwin):
        """Bring ``subwin`` to the foreground, restoring if minimized."""
        if subwin.isMinimized():
            subwin.showNormal()
        self._mgr.mdiArea.setActiveSubWindow(subwin)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
