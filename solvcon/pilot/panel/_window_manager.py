# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Window manager feature for pilot.

Arrange the open MDI sub-windows from the "Window" menu, list them
there, and bring one to the foreground when its entry is chosen.
"""

from PySide6 import QtGui, QtWidgets

from ..base import _gui_common

__all__ = [
    'WindowManager',
]


class WindowManager(_gui_common.PilotFeature):
    """Arrange and list open MDI sub-windows under the "Window" menu.

    The menu carries three sections: the arrangement actions, the tabbed
    view toggle, and one checkable action per open sub-window labelled by
    its title. Triggering a sub-window entry activates it, and the active
    one is checked. Every section is refreshed when the menu is about to
    show.
    """

    #: objectName tagging every dynamic per-sub-window action.
    ITEM_ID = "window.subwindow"
    #: objectName of the tile layout action.
    TILE_ID = "window.layout.tile"
    #: objectName of the cascade layout action.
    CASCADE_ID = "window.layout.cascade"
    #: objectName of the single-row tiling action.
    HORIZONTAL_ID = "window.layout.horizontal"
    #: objectName of the single-column tiling action.
    VERTICAL_ID = "window.layout.vertical"
    #: objectName of the tabbed view mode toggle.
    TABBED_ID = "window.layout.tabbed"

    def __init__(self, *args, **kw):
        super(WindowManager, self).__init__(*args, **kw)
        self._menu = None
        self._items = []
        self._layout_actions = []

    def populate_menu(self):
        """Build the static sections and anchor the dynamic list.

        Entries are placed by weight, leaving room for a later feature to
        slot one in. The dynamic list is seeded right away: a native menu
        bar hides an empty menu, and a hidden menu never fires
        aboutToShow to fill itself.
        """
        self._menu = self._mgr.menu_model.menu("Window")

        self._layout_actions = [
            self.add_action(
                "Window", "Tile",
                "Tile the open sub-windows to fill the area",
                self._tile, id=self.TILE_ID, weight=10),
            self.add_action(
                "Window", "Tile Horizontally",
                "Arrange the open sub-windows in a single row",
                self._tile_horizontal, id=self.HORIZONTAL_ID, weight=11),
            self.add_action(
                "Window", "Tile Vertically",
                "Arrange the open sub-windows in a single column",
                self._tile_vertical, id=self.VERTICAL_ID, weight=12),
            self.add_action(
                "Window", "Cascade",
                "Stack the open sub-windows with an offset",
                self._cascade, id=self.CASCADE_ID, weight=13),
        ]
        self._mgr.menu_model.place_separator("Window", weight=20)

        act = self.add_action(
            "Window", "Tabbed View",
            "Show the sub-windows as pages of a tab bar",
            None, id=self.TABBED_ID, weight=30, checkable=True)
        act.toggled.connect(self._set_tabbed)

        self._mgr.menu_model.place_separator("Window", weight=40)

        self._menu.aboutToShow.connect(self._rebuild)
        self._rebuild()

    def _tile(self):
        self._mgr.mdiArea.tileSubWindows()

    def _cascade(self):
        self._mgr.mdiArea.cascadeSubWindows()

    def _set_tabbed(self, on):
        """Switch the MDI area between sub-window and tabbed view.

        Closable and movable tabs keep the window controls that the
        sub-window frames provided.
        """
        mdi = self._mgr.mdiArea
        if on:
            mdi.setViewMode(QtWidgets.QMdiArea.ViewMode.TabbedView)
            mdi.setTabsClosable(True)
            mdi.setTabsMovable(True)
        else:
            mdi.setViewMode(QtWidgets.QMdiArea.ViewMode.SubWindowView)
        self._update_layout_actions()

    def _tile_horizontal(self):
        self._arrange(horizontal=True)

    def _tile_vertical(self):
        self._arrange(horizontal=False)

    def _update_layout_actions(self):
        """Enable the arrangement actions only when they can act.

        They need a sub-window to arrange and the sub-window view to
        arrange it in; geometry means nothing while the area shows tabs.
        """
        mdi = self._mgr.mdiArea
        subwins = [s for s in mdi.subWindowList() if s.isVisible()]
        tabbed = mdi.viewMode() == QtWidgets.QMdiArea.ViewMode.TabbedView
        for act in self._layout_actions:
            act.setEnabled(bool(subwins) and not tabbed)

    def _arrange(self, horizontal):
        """Line the visible sub-windows up along one direction.

        Each takes an equal share of the viewport: a row over the full
        height when ``horizontal``, a column over the full width
        otherwise. ``QMdiArea`` offers no directional counterpart to
        ``tileSubWindows``, so the geometry is dealt out by hand. A
        minimized or maximized sub-window is restored first, or its new
        geometry would not take effect.
        """
        mdi = self._mgr.mdiArea
        subwins = [s for s in mdi.subWindowList() if s.isVisible()]
        if not subwins:
            return

        area = mdi.contentsRect()
        width, height = area.width(), area.height()
        if horizontal:
            width //= len(subwins)
        else:
            height //= len(subwins)

        for index, subwin in enumerate(subwins):
            if subwin.isMinimized() or subwin.isMaximized():
                subwin.showNormal()
            x = area.x() + index * width if horizontal else area.x()
            y = area.y() if horizontal else area.y() + index * height
            subwin.setGeometry(x, y, width, height)

    def _rebuild(self):
        """Refresh the menu to match the MDI area.

        Drop the entries from the previous show, then append one checkable
        action per visible sub-window in area order, checking the active
        one. A disabled placeholder stands in when none are open.
        """
        for act in self._items:
            self._menu.removeAction(act)
            act.deleteLater()
        self._items = []

        mdi = self._mgr.mdiArea
        active = mdi.activeSubWindow()
        subwins = [s for s in mdi.subWindowList() if s.isVisible()]

        self._update_layout_actions()

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
