# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import contextlib
import signal

from .. import _pilot_core as _pcore
from ._detach import SubWindowDetacher
from ...core import Toggle
from PySide6 import QtCore, QtGui


# Milliseconds between the wake-ups that let the SIGINT handler run.
INTERRUPT_INTERVAL = 200


@contextlib.contextmanager
def interrupt_exits(stop=QtCore.QCoreApplication.exit,
                    interval=INTERRUPT_INTERVAL):
    """Let Ctrl-C end the Qt event loop run inside the block.

    Python runs a signal handler only between bytecodes, and an idle Qt loop
    runs none, so a timer wakes the interpreter every ``interval``
    milliseconds to let the handler fire. The handler queues
    ``stop(128 + SIGINT)``, the status a shell expects of a program stopped
    by Ctrl-C. It queues rather than calls, because the default stop runs
    the ``aboutToQuit`` hooks at once, and the handler may interrupt any
    callback. Ctrl-C during a console command also stops the command, and
    the loop ends when it returns.

    :param stop: Callable that ends the loop with a status. The default,
        :meth:`QCoreApplication.exit`, leaves the thread unable to run
        another event loop until the application loop starts again.
    :param interval: Milliseconds between wake-ups, the longest Ctrl-C
        waits for the handler while the loop idles.
    """
    def _handle(signum, frame):
        QtCore.QTimer.singleShot(0, lambda: stop(128 + signum))

    previous = signal.signal(signal.SIGINT, _handle)
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(interval)
    try:
        yield
    finally:
        timer.stop()
        signal.signal(signal.SIGINT, previous)


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


_message_sink = None
_pending_messages = []


def write_message(text):
    """Show ``text`` to the user in the pilot's command transcript.

    A message written before the transcript exists waits for it, so what a
    feature reports while the pilot starts is not lost.
    """
    if _message_sink is None:
        _pending_messages.append(text)
    else:
        _message_sink(text)


def set_message_sink(sink):
    """Send every message to ``sink``, starting with the ones waiting for
    it.  ``None`` holds the messages again."""
    global _message_sink
    _message_sink = sink
    if sink is not None:
        pending = list(_pending_messages)
        del _pending_messages[:]
        for text in pending:
            sink(text)


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


class SubWindowCloseFilter(QtCore.QObject):
    """Report a watched sub-window's close synchronously.

    A ``QMdiSubWindow`` has no close signal, so this filter is the only way
    for an owner to drop its cached widget reference before Qt frees the
    widget the sub-window wraps (``Qt.WA_DeleteOnClose`` runs the delete on
    the next event-loop turn, not inside ``close()`` itself).
    """

    def __init__(self, on_close, parent):
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Close:
            self._on_close()
        return False


def keep_filling(subwindow, area):
    """Keep ``subwindow`` filling ``area`` when the area takes it out of
    maximized.

    The area keeps one sub-window maximized: bringing another to the front
    puts the maximized one back to its normal size, which for a window
    opened maximized is the small size it was added with.  The window
    then fills the area instead, at its size of the moment.  A restore of
    the window in front, which is the user's, keeps its normal size.

    :param subwindow: The ``QMdiSubWindow`` to keep.
    :param area: The ``QMdiArea`` holding it, from the manager.
    """
    def fill():
        state = subwindow.windowState()
        if (not state & (QtCore.Qt.WindowMaximized
                         | QtCore.Qt.WindowMinimized)
                and area.activeSubWindow() is not subwindow):
            subwindow.setGeometry(area.viewport().rect())

    def on_state(old, new):
        # The area takes the window out of maximized while it is still in
        # front, and brings the other forward right after; who is in front
        # then tells the two moves apart.
        if (old & QtCore.Qt.WindowMaximized
                and not new & QtCore.Qt.WindowMaximized):
            QtCore.QTimer.singleShot(0, subwindow, fill)
    subwindow.windowStateChanged.connect(on_state)


def setup_app_subwindow(subwindow, mgr, app_config):
    """Give the sub-window of an app what every app window has: the size
    grip, a detacher that moves it out of the main window and back, the
    fill of the area when another window comes to the front, and its
    config shown when it comes to the front out of the area.

    :param subwindow: The ``QMdiSubWindow`` the manager added for the app.
    :param mgr: The ``RManager`` holding it.
    :param app_config: The ``AppConfigPanel`` the app's column is in.
    :return: The :class:`SubWindowDetacher` of the sub-window.
    """
    mgr.addSubWindowGrip(subwindow)
    detacher = SubWindowDetacher(subwindow, mgr)
    keep_filling(subwindow, mgr.mdiArea)
    # The detacher goes with the sub-window, and so does this connection.
    detacher.activated.connect(lambda: app_config.show_for(subwindow))
    return detacher


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
        _pcore.require('a pilot shortcut')
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
        _pcore.require(type(self).__name__)
        if not isinstance(self._mgr, _pcore.RManager):
            raise TypeError(
                "'mgr' must be an instance of 'solvcon.pilot.RManager'")
        super(PilotFeature, self).__init__(*args, **kw)

    @staticmethod
    def _write_message(text):
        write_message(text)

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
