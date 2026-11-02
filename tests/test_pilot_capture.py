# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for screenshots and movies of the pilot.
"""

import os
import platform
import tempfile
import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot import base
    from solvcon.pilot.base import _capture
    from PySide6.QtCore import QObject, QSize, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QDockWidget, QLabel

    class _Source(QObject):
        changed = Signal(str)

    class _Blank(QLabel):
        def grab(self, *args):
            return QPixmap()
except ImportError:
    pilot = None
    _capture = None


@testing.requires_pilot
@unittest.skipIf("Windows" == platform.system(),
                 "offscreen grabbing is unreliable on Windows CI")
class WidgetCaptureTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = os.path.join(self._folder.name, "shots")

    def test_shot_saves_the_widget_whole(self):
        widget = QLabel("shock tube")
        widget.resize(160, 90)
        path = _capture.WidgetCapture(self.folder).shot(widget, "label")
        self.assertEqual(os.path.join(self.folder, "label.png"), path)
        ratio = widget.grab().devicePixelRatio()
        self.assertEqual(QSize(round(160 * ratio), round(90 * ratio)),
                         QImage(path).size())

    def test_shot_shows_a_queued_update(self):
        label = QLabel("before")
        source = _Source()
        source.changed.connect(label.setText, Qt.QueuedConnection)
        source.changed.emit("after")
        self.assertEqual("before", label.text())
        _capture.WidgetCapture(self.folder).shot(label, "label")
        self.assertEqual("after", label.text())

    def test_next_name_never_reuses_a_file(self):
        capture = _capture.WidgetCapture(self.folder)
        capture.shot(QLabel("a"), "main_01")
        capture.shot(QLabel("b"), "main")
        self.assertEqual("main_02",
                         capture.next_name("{target}_{n:02}", target="main"))
        self.assertEqual("main_1", capture.next_name("main"))

    def test_shot_takes_a_name_not_a_path(self):
        capture = _capture.WidgetCapture(self.folder)
        for name in ("", os.path.join("sub", "label")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                capture.shot(QLabel("x"), name)


@testing.requires_pilot
@unittest.skipIf("Windows" == platform.system(),
                 "offscreen grabbing is unreliable on Windows CI")
class ScreenCaptureTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.capture = base.ScreenCapture(mgr=self.mgr)
        self.capture.folder = self.folder

    def test_shoot_names_the_file_after_its_target(self):
        dock = QDockWidget("Shot Test")
        dock.setWidget(QLabel("dock"))
        self.mgr.mainWindow.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.addCleanup(self.mgr.mainWindow.removeDockWidget, dock)
        label = QLabel("widget")
        label.setObjectName("probe")
        paths = [self.capture.shoot(), self.capture.shoot('Shot Test'),
                 self.capture.shoot(label), self.capture.shoot('main')]
        self.assertEqual(['main_01.png', 'shot_test_01.png', 'probe_01.png',
                          'main_02.png'],
                         [os.path.basename(p) for p in paths])

    def test_a_region_keeps_only_that_part(self):
        label = QLabel("region")
        label.resize(200, 100)
        ratio = label.grab().devicePixelRatio()
        path = self.capture.shoot(label, region=(10, 20, 60, 30))
        self.assertEqual((round(60 * ratio), round(30 * ratio)),
                         (QImage(path).width(), QImage(path).height()))

    def test_a_recording_writes_its_movie_once(self):
        label = QLabel("frames")
        label.resize(64, 48)
        recording = self.capture.record(label)
        path = recording.stop()
        self.assertTrue(os.path.isfile(path))
        self.assertIsNone(self.capture.recording)
        self.assertIsNone(recording.stop())

    def test_a_recording_that_gets_no_frame_ends(self):
        self.capture.record(_Blank("blank"))
        self.assertIsNone(self.capture.recording)
        self.assertIsNone(self.capture.stop_recording())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
