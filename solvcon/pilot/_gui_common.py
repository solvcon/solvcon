# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from . import _pilot_core as _pcore
from PySide6 import QtCore, QtGui


def build_action(parent, text, tip, func, *, id=None, checkable=False,
                 checked=False, shortcut=None, menu_role=None):
    """Build a QAction from one description, the single item builder.

    The id becomes the action's objectName (its handle in the menu model and
    in tests). A checkable action carries its initial check state; ``func``,
    when given, runs on trigger. A checkable feature that needs the toggled
    state wires ``toggled`` on the returned action itself.
    """
    act = QtGui.QAction(text, parent)
    if id:
        act.setObjectName(id)
    act.setStatusTip(tip)
    if shortcut is not None:
        act.setShortcut(shortcut)
    if menu_role is not None:
        act.setMenuRole(menu_role)
    if checkable:
        act.setCheckable(True)
        act.setChecked(checked)
    if func is not None:
        act.triggered.connect(lambda *a: func())
    return act


class PilotFeature(QtCore.QObject):
    """
    Base class to house common GUI code for prototyping pilot features.

    :ivar _mgr:
        The solvcon pilot application manager implemented with Qt in C++.
    :vartype mgr: solvcon.pilot.RManager
    """

    def __init__(self, *args, **kw):
        """
        :param mgr:
            The solvcon pilot application manager implemented with Qt in C++.
        :type mgr: solvcon.pilot.RManager
        """
        self._mgr = kw.pop('mgr')
        if not isinstance(self._mgr, _pcore.RManager):
            raise TypeError(
                "'mgr' must be an instance of 'solvcon.pilot.RManager'")
        super(PilotFeature, self).__init__(*args, **kw)

    @property
    def _pycon(self):
        """
        :rtype: solvcon.pilot.RPythonConsoleDockWidget
        """
        return self._mgr.pycon

    @property
    def _mainWindow(self):
        """
        :rtype: PySide6.QtWidget.QMainWindow
        """
        return self._mgr.mainWindow

    def add_action(self, path, text, tip, func, *, id, weight=50,
                   checkable=False, checked=False, shortcut=None,
                   menu_role=None):
        """Build an action and place it in the menu at ``path`` by ``weight``.

        This is the one Python item builder over the shared placement: it
        returns the live action so a toggle feature can wire its own reverse
        connections.
        """
        act = build_action(self._mainWindow, text, tip, func, id=id,
                           checkable=checkable, checked=checked,
                           shortcut=shortcut, menu_role=menu_role)
        self._mgr.menu_model.place(path, act, weight)
        return act

    def _add_menu_item(self, menu, text, tip, func):
        """
        Add an item to the corresponding menu.

        :param menu: The menu to add the item to.
        :type menu: PySide6.QtWidget.QMenu
        :param text: Menu description string.
        :param tip: Menu tip string.
        :param func: Python callable.

        :return: None
        """
        menu.addAction(build_action(self._mainWindow, text, tip, func))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
