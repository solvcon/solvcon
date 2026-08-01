# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Keep the pilot GUI tests off the screen and out of the user's configuration.

The GUI tests drive real top-level windows, and every one that maps takes the
desktop and the keyboard away from whoever is running the suite. Marking a
window Qt::WA_DontShowOnScreen as it is shown leaves it created, laid out, and
painted into its backing store while it never reaches the screen, so the tests
keep running against a live window surface. That is what the offscreen QPA
platform cannot offer: the suite skips its window tests there. Export
``SOLVCON_TEST_SHOW_WINDOWS=ON`` to watch the windows instead, which hands them
the desktop and the keyboard for the length of the run.

The suite also points ``SOLVCON_CONFIG_HOME`` at a scratch directory, so a test
that reads or writes the application configuration neither picks up the
settings of whoever is running it nor overwrites them.
"""


import os
import shutil
import tempfile

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


_config_home = None
_saved_config_home = None


def pytest_configure(config):
    """Send the application configuration to a scratch directory for the run.

    Set before any test imports the pilot, which reads the configuration as it
    builds its panels.
    """
    global _config_home, _saved_config_home
    _saved_config_home = os.environ.get("SOLVCON_CONFIG_HOME")
    _config_home = tempfile.mkdtemp(prefix="solvcon-test-config-")
    os.environ["SOLVCON_CONFIG_HOME"] = _config_home


def pytest_unconfigure(config):
    """Remove the directory this file created, and only that one.

    Deleting whatever the variable happens to point at would follow a test
    that repointed it and failed before restoring, and recursively delete a
    real directory such as the developer's own configuration.
    """
    if _saved_config_home is None:
        os.environ.pop("SOLVCON_CONFIG_HOME", None)
    else:
        os.environ["SOLVCON_CONFIG_HOME"] = _saved_config_home
    if _config_home:
        shutil.rmtree(_config_home, ignore_errors=True)


def pytest_collection_finish(session):
    _hide_windows()


def pytest_runtest_setup(item):
    _hide_windows()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
