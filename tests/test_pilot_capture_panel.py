# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the Capture dock.
"""

import platform
import tempfile
import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot import base, panel
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
except ImportError:
    pilot = None


@testing.requires_pilot
@unittest.skipIf("Windows" == platform.system(),
                 "offscreen grabbing is unreliable on Windows CI")
class CaptureWidgetTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.capture = base.ScreenCapture(mgr=self.mgr)
        self.capture.folder = self.folder

    def test_the_dock_follows_the_capture(self):
        widget = panel.CaptureWidget(mgr=self.mgr, capture=self.capture)
        widget._ensure_dock()
        self.addCleanup(self.mgr.mainWindow.removeDockWidget, widget._dock)
        column = widget._column

        path = self.capture.shoot('main')
        self.assertEqual(path, column._selected_path())

        label = QLabel("frames")
        label.resize(64, 48)
        with self.capture.record(label, suffix='.webp') as recording:
            self.assertTrue(column._record.text().startswith("Stop"))
            self.assertFalse(column._format.isEnabled())
        self.assertEqual("Record", column._record.text())
        self.assertTrue(column._format.isEnabled())
        self.assertEqual(recording.path,
                         column._shots.item(0).data(Qt.UserRole))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
