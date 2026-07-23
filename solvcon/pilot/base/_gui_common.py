# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from .. import _pilot_core as _pcore
from ...core import Toggle
from PySide6 import QtCore, QtGui


def apply_label_mode(overlay, on, advanced, coordinates):
    """Pack the shape-label switch, its mode, and the coordinate switch into
    an overlay's flags. The shape labels and the grid coordinate labels are
    independent, so each switch owns its own flags."""
    overlay.shape_ids = on and not advanced
    overlay.advanced_labels = on and advanced
    overlay.coordinate_labels = coordinates
    return overlay


def label_switch_and_mode(overlay):
    """Return ``(on, advanced, coordinates)`` for an overlay's label state."""
    on = overlay.shape_ids or overlay.advanced_labels
    return on, overlay.advanced_labels, overlay.coordinate_labels


class ToggleActionBridge(QtCore.QObject):
    """Two-way binding between a checkable QAction and a store toggle.

    The action's ``toggled`` writes the toggle; the toggle's ``on_change``
    re-checks the action. The change is delivered on the UI thread through a
    queued signal, so a change from a solver thread is safe. The store's no-op
    guard (a set to the value already stored fires no ``on_change``) plus Qt
    emitting ``toggled`` only on a real change together stop the echo, so the
    old ``blockSignals`` guard is not needed.

    The bridge is parented to its action, so it and its subscription live
    exactly as long as the action.
    """

    _store_changed = QtCore.Signal(bool)

    def __init__(self, action, key):
        super().__init__(action)
        self._action = action
        self._key = key
        tg = Toggle.instance
        tg.declare_bool(key, action.isChecked())
        action.setChecked(tg.get(key, action.isChecked()))
        action.toggled.connect(self._on_action_toggled)
        # Deliver a store change to the UI thread, then re-check the action.
        self._store_changed.connect(action.setChecked,
                                    QtCore.Qt.QueuedConnection)
        self._token = tg.on_change(key, self._on_store_changed)
        # Drop the subscription when the action goes away, so a later store
        # change never calls back into a destroyed widget.
        action.destroyed.connect(self._release)

    def _release(self, *args):
        self._token = None

    def _on_action_toggled(self, checked):
        if self._token is None:
            return
        Toggle.instance.set_bool(self._key, checked)

    def _on_store_changed(self):
        if self._token is None:
            return
        tg = Toggle.instance
        self._store_changed.emit(tg.get(self._key, self._action.isChecked()))


_SHORTCUT_CONTEXTS = {
    "application": QtCore.Qt.ApplicationShortcut,
    "window": QtCore.Qt.WindowShortcut,
    "widget": QtCore.Qt.WidgetShortcut,
}

_MENU_ROLES = {
    "none": QtGui.QAction.NoRole,
    "quit": QtGui.QAction.QuitRole,
    "preferences": QtGui.QAction.PreferencesRole,
    "about": QtGui.QAction.AboutRole,
}


def apply_shortcut(action, mgr=None):
    """Apply the keymap binding for ``action``'s objectName from the roof.

    Resolves the command id through ``RManager.resolve_shortcut`` and sets the
    menu role, shortcut context, and sequences with PySide. Unknown ids are a
    no-op so feature actions outside the vocabulary stay untouched. The role is
    set first, as macOS requires before the action reaches a menu.
    """
    oid = action.objectName()
    if not oid:
        return
    if mgr is None:
        mgr = _pcore.RManager.instance
    resolved = mgr.resolve_shortcut(oid)
    if not resolved["known"]:
        return

    action.setMenuRole(_MENU_ROLES[resolved["role"]])
    action.setShortcutContext(_SHORTCUT_CONTEXTS[resolved["context"]])
    if resolved["standard"]:
        standard = getattr(QtGui.QKeySequence.StandardKey,
                           resolved["standard_key"])
        action.setShortcuts(standard)
    else:
        action.setShortcuts([
            QtGui.QKeySequence(s, QtGui.QKeySequence.PortableText)
            for s in resolved["sequences"]
        ])


def build_action(parent, text, tip, func, *, id=None, checkable=False,
                 checked=False, shortcut=None, menu_role=None,
                 toggle_key=None):
    """Build a QAction from one description, the single item builder.

    The id becomes the action's objectName (its handle in the menu model and
    in tests). A checkable action carries its initial check state; ``func``,
    when given, runs on trigger. A checkable feature that needs the toggled
    state wires ``toggled`` on the returned action itself.

    When ``toggle_key`` is given, the action is bound two-way to that store
    toggle through a ToggleActionBridge, so its state persists, round-trips to
    JSON, and can be observed by a solver or a second widget.

    When ``id`` names a keymap command, ``apply_shortcut`` installs its
    platform binding and overwrites any explicit ``shortcut`` or ``menu_role``.
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
    if toggle_key is not None:
        ToggleActionBridge(act, toggle_key)
    if id:
        apply_shortcut(act)
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
                   menu_role=None, toggle_key=None):
        """Build an action and place it in the menu at ``path`` by ``weight``.

        This is the one Python item builder over the shared placement: it
        returns the live action so a toggle feature can wire its own reverse
        connections. When ``toggle_key`` is given, the action is bound two-way
        to that store toggle (see build_action), so RMenuModel stays the
        placement layer and the Toggle store owns the value.
        """
        act = build_action(self._mainWindow, text, tip, func, id=id,
                           checkable=checkable, checked=checked,
                           shortcut=shortcut, menu_role=menu_role,
                           toggle_key=toggle_key)
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
