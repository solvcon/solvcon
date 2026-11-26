# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Move the widget of an MDI sub-window into a window of its own and back.
"""

import functools

from PySide6 import QtCore, QtWidgets

__all__ = [
    'DetachedWindow',
    'SubWindowDetacher',
]

# The Qt objects below outlive their owners' references until Qt deletes
# them, and a wrapper collected before that would leave their overrides to
# run on a new, empty one.  A sub-window from the manager reaches Python
# without its ownership, so the parent does not hold the wrapper.
_alive = set()


def _keep(obj):
    """Hold the wrapper of ``obj`` until Qt destroys the object."""
    _alive.add(obj)
    obj.destroyed.connect(functools.partial(_alive.discard, obj))


class DetachedWindow(QtWidgets.QWidget):
    """The window of its own that holds a detached widget.

    It is a top-level window owned by the main window, so it goes away
    with the pilot.  Closing it moves the widget back into the main
    window rather than closing it.

    :param detacher: The :class:`SubWindowDetacher` that made it.
    :param parent: The main window.
    """

    def __init__(self, detacher, parent):
        super().__init__(parent, QtCore.Qt.Window)
        #: The :class:`SubWindowDetacher` that made the window.
        self.detacher = detacher
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        _keep(self)

    def changeEvent(self, event):
        if (QtCore.QEvent.ActivationChange == event.type()
                and self.isActiveWindow()):
            self.detacher.activated.emit()
        super().changeEvent(event)

    def closeEvent(self, event):
        # The attach hides and deletes this window.
        self.detacher.attach()
        event.ignore()


class SubWindowDetacher(QtCore.QObject):
    """Let the widget of ``subwindow`` move into a window of its own and
    back.

    The sub-window stays, hidden, while its widget is away, so whoever
    keys anything on the sub-window, such as the "App config" panel or a
    close filter, keeps working.  The detacher is a child of the
    sub-window and goes away with it; :meth:`of` finds it.

    :param subwindow: The ``QMdiSubWindow`` whose widget moves.
    :param mgr: The pilot manager, whose main window owns the window of
        its own and whose area holds the sub-window.
    """

    #: The window of its own became the active window.
    activated = QtCore.Signal()

    def __init__(self, subwindow, mgr):
        super().__init__(subwindow)
        #: The ``QMdiSubWindow`` whose widget moves.
        self.subwindow = subwindow
        #: The :class:`DetachedWindow` while the widget is out, or None.
        self.window = None
        # The area is reached through the manager: a wrapper of it from
        # the sub-window would delete the area when collected.
        self._mgr = mgr
        self._was_maximized = False
        menu = subwindow.systemMenu()
        if menu is not None:
            menu.addSeparator()
            menu.addAction("Detach to Own Window", self.detach)
        subwindow.windowTitleChanged.connect(self._on_title)
        subwindow.installEventFilter(self)
        _keep(self)

    @staticmethod
    def of(subwindow):
        """Return the detacher of ``subwindow``, or None.

        :param subwindow: A ``QMdiSubWindow``, or None.
        """
        if subwindow is None:
            return None
        return subwindow.findChild(SubWindowDetacher, None,
                                   QtCore.Qt.FindDirectChildrenOnly)

    @property
    def detached(self):
        """Whether the widget is in a window of its own."""
        return self.window is not None

    def detach(self):
        """Move the widget into a window of its own the size and place of
        the sub-window."""
        widget = self.subwindow.widget()
        if self.detached or widget is None:
            return
        self._was_maximized = self.subwindow.isMaximized()
        corner = self.subwindow.mapToGlobal(QtCore.QPoint(0, 0))
        size = self.subwindow.size()
        self.window = DetachedWindow(self, self._mgr.mainWindow)
        self.window.setWindowTitle(self.subwindow.windowTitle())
        self.subwindow.setWidget(None)
        self.window.layout().addWidget(widget)
        self.subwindow.hide()
        self.window.setGeometry(QtCore.QRect(corner, size))
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        widget.setFocus()

    def attach(self):
        """Move the widget back into the sub-window, maximized if it was."""
        if not self.detached:
            return
        self._take_back()
        if self._was_maximized:
            self.subwindow.showMaximized()
        else:
            self.subwindow.show()
        self._mgr.mdiArea.setActiveSubWindow(self.subwindow)
        self.subwindow.widget().setFocus()

    def toggle(self):
        """Detach the widget, or attach it when it is out."""
        if self.detached:
            self.attach()
        else:
            self.detach()

    def bring_forward(self):
        """Show the widget in front, wherever it is."""
        if self.detached:
            if self.window.isMinimized():
                self.window.showNormal()
            self.window.raise_()
            self.window.activateWindow()
            return
        self._mgr.mdiArea.setActiveSubWindow(self.subwindow)
        self.subwindow.showMaximized()

    def eventFilter(self, obj, event):
        # A widget still out goes with its own window rather than back into
        # the closing sub-window: a domain view moved into a window that is
        # never shown again would outlive the graphics of the window it
        # left, and crash when deleted.
        if (obj is self.subwindow and self.detached
                and QtCore.QEvent.Close == event.type()):
            window, self.window = self.window, None
            window.hide()
            window.deleteLater()
        return False

    def _take_back(self):
        window, self.window = self.window, None
        widget = window.layout().itemAt(0).widget()
        window.layout().removeWidget(widget)
        self.subwindow.setWidget(widget)
        window.hide()
        window.deleteLater()

    def _on_title(self, title):
        if self.window is not None:
            self.window.setWindowTitle(title)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
