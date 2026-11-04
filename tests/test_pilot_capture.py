# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for screenshots and movies of the pilot.
"""

import os
import platform
import tempfile
import unittest

from solvcon import apputil, testing

try:
    from solvcon import pilot
    from solvcon.pilot import base
    from solvcon.pilot.base import _capture
    from PySide6.QtCore import QEventLoop, QObject, QSize, Qt, QTimer, Signal
    from PySide6.QtGui import QImage, QPainter, QPixmap
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
class LogicalSizeTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_a_high_dpi_grab_comes_down_to_the_widget_size(self):
        pixmap = QPixmap(320, 180)
        pixmap.fill(Qt.GlobalColor.darkGreen)
        pixmap.setDevicePixelRatio(2.0)
        come_down = _capture.to_logical_size(pixmap)
        self.assertEqual(QSize(160, 90), come_down.size())
        self.assertEqual(1.0, come_down.devicePixelRatio())


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

        capture = _capture.WidgetCapture(self.folder, logical=True)
        path = capture.shot(widget, "logical")
        self.assertEqual(QSize(160, 90), QImage(path).size())

    def test_next_name_moves_on_from_a_pattern_with_no_counter(self):
        os.makedirs(self.folder)
        open(os.path.join(self.folder, "{n}.png"), "w").close()
        capture = _capture.WidgetCapture(self.folder)
        self.assertEqual("{n}_1", capture.next_name("{{n}}"))

    def test_shot_shows_a_queued_update(self):
        label = QLabel("before")
        source = _Source()
        source.changed.connect(label.setText, Qt.QueuedConnection)
        source.changed.emit("after")
        self.assertEqual("before", label.text())
        _capture.WidgetCapture(self.folder).shot(label, "label")
        self.assertEqual("after", label.text())

    def test_trim_cuts_the_plain_edges_down_to_the_content(self):
        pixmap = QPixmap(40, 100)
        pixmap.fill(Qt.GlobalColor.lightGray)
        painter = QPainter(pixmap)
        painter.fillRect(5, 10, 20, 20, Qt.GlobalColor.black)
        painter.end()
        trimmed = _capture.trim_edges(pixmap)
        self.assertEqual(QSize(20, 20), trimmed.size())

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
                 self.capture.shoot(label), self.capture.shoot('main'),
                 self.capture.shoot(dock)]
        self.assertEqual(['main_01.png', 'shot_test_01.png', 'probe_01.png',
                          'main_02.png', 'shot_test_02.png'],
                         [os.path.basename(p) for p in paths])

    def test_a_bad_name_pattern_is_a_value_error(self):
        for pattern in ('{nope}', '{target.nope}', '{n[0]}'):
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                self.capture.shoot(name=pattern)

    def test_delayed_shots_do_not_share_a_name(self):
        label = QLabel("widget")
        first = self.capture.shoot(label, delay=0.01)
        second = self.capture.shoot(label, delay=0.01)
        self.assertNotEqual(first, second)
        self.assertFalse(os.path.exists(first))
        loop = QEventLoop()
        QTimer.singleShot(200, loop.quit)
        loop.exec()
        self.assertEqual({first, second}, set(self.capture.shots))
        self.assertTrue(os.path.isfile(second))

    def test_console_gets_the_capture_helpers(self):
        env = apputil.AppEnvironment('capture_console')
        banner = apputil.install_pilot_namespace(
            self.mgr, env, extra=self.capture.console_handles())
        self.assertIn('capture(target)', banner)
        env.run_code("path = capture('main', 'console')")
        path = os.path.join(self.folder, 'console.png')
        self.assertEqual(path, env.globals['path'])
        self.assertTrue(os.path.isfile(path))

    def test_a_bad_name_pattern_is_reported(self):
        reports = []
        self.capture.reported.connect(lambda _, ok: reports.append(ok))
        for pattern in ('{nope}', '{target.nope}', '{n[0]}'):
            self.capture.pattern = pattern
            self.assertIsNone(self.capture.shoot_reporting())
        self.assertEqual([False, False, False], reports)

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

        reports = []
        self.capture.reported.connect(lambda _, ok: reports.append(ok))
        self.capture.record_reporting(_Blank("blank"))
        self.assertEqual([False], reports)

    def test_a_with_block_writes_the_movie(self):
        label = QLabel("frames")
        label.resize(64, 48)
        with self.capture.record(label, 'clip', suffix='.webp'):
            path = os.path.join(self.folder, 'clip.webp')
            self.assertFalse(os.path.exists(path))
        self.assertTrue(os.path.isfile(path))
        self.assertEqual([path], self.capture.shots)

    def test_a_movie_refuses_what_it_cannot_write(self):
        label = QLabel("frames")
        for kw in ({'suffix': '.avi'}, {'every_ms': 0}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.capture.record(label, **kw)
        self.assertIsNone(self.capture.recording)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
