# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.base import _gui_common
    from PySide6 import QtCore, QtWidgets
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ToggleActionBridgeTC(unittest.TestCase):

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.tg = solvcon.Toggle.instance
        # A unique key per test isolates the shared global store.
        self.KEY = "pilot.test.toggle_action." + self._testMethodName
        self.tg.declare_bool(self.KEY, False)
        self.tg.set_bool(self.KEY, False)
        self.act = _gui_common.build_action(
            self.mgr.mainWindow, "Flag", "A test toggle", None,
            id="test.flag", checkable=True, checked=False,
            toggle_key=self.KEY)

    def tearDown(self):
        # Destroy the action so its bridge unsubscribes from the store.
        self.act.deleteLater()
        self.act = None
        QtWidgets.QApplication.sendPostedEvents(
            None, QtCore.QEvent.Type.DeferredDelete)

    def _drain(self):
        QtWidgets.QApplication.processEvents()

    def test_declares_toggle_with_default(self):
        self.assertEqual(self.tg.get(self.KEY, True), False)

    def test_menu_toggle_writes_the_store(self):
        self.act.setChecked(True)
        self.assertEqual(self.tg.get(self.KEY, False), True)
        self.act.setChecked(False)
        self.assertEqual(self.tg.get(self.KEY, True), False)

    def test_store_set_rechecks_without_echo(self):
        # Count notifications independently: a store-side change must fire
        # on_change exactly once. The re-check it triggers writes the same
        # value back, which the no-op guard drops, so there is no echo.
        fires = []
        token = self.tg.on_change(self.KEY, lambda: fires.append(1))

        self.act.setChecked(False)  # ensure a real change below
        fires.clear()

        self.tg.set_bool(self.KEY, True)
        self._drain()  # deliver the queued setChecked on the UI thread

        self.assertTrue(self.act.isChecked())
        self.assertEqual(len(fires), 1)
        del token


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
