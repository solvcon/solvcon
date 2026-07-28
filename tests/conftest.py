# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Keep the pilot GUI tests off the screen.

The GUI tests drive real top-level windows, and every one that maps takes the
desktop and the keyboard away from whoever is running the suite. Marking a
window Qt::WA_DontShowOnScreen as it is shown leaves it created, laid out, and
painted into its backing store while it never reaches the screen, so the tests
keep running against a live window surface. That is what the offscreen QPA
platform cannot offer: the suite skips its window tests there. Export
``SOLVCON_TEST_SHOW_WINDOWS=ON`` to watch the windows instead, which hands them
the desktop and the keyboard for the length of the run.
"""


import os

try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    QtCore = QtWidgets = None

SHOW_WINDOWS = (os.getenv('SOLVCON_TEST_SHOW_WINDOWS') or '').upper() in (
    '1', 'ON', 'YES', 'TRUE')

_hider = None


def _build_hider():
    class WindowHider(QtCore.QObject):
        def eventFilter(self, obj, event):
            if (QtCore.QEvent.Type.Show == event.type()
                    and isinstance(obj, QtWidgets.QWidget)
                    and obj.isWindow()):
                obj.setAttribute(
                    QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            return False

    return WindowHider()


def _hide_windows():
    """Install the window hider on the Qt application once it exists.

    Importing the pilot test modules is what creates the application, so the
    earliest hook this can run from is the end of collection. Retry from every
    test setup to cover an application that first appears later.
    """
    global _hider
    if _hider is not None or SHOW_WINDOWS or QtCore is None:
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    _hider = _build_hider()
    app.installEventFilter(_hider)


def pytest_collection_finish(session):
    _hide_windows()


def pytest_runtest_setup(item):
    _hide_windows()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
