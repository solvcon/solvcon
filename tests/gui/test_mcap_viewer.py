# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The MCAP panel feature on the pilot window: the menu and the dock."""

import os
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.track import _mcap_viewer
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError:
    pilot = None

try:
    from mcap import writer as foxglove_mcap_writer
except ImportError:
    foxglove_mcap_writer = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))

TOPIC = "/vehicle/brake"


def write_fixture(path):
    """One schemaless topic: the feature tests only list it."""
    with open(path, "wb") as fp:
        writer = foxglove_mcap_writer.Writer(fp)
        writer.start(profile="ros2")
        channel_id = writer.register_channel(TOPIC, "cdr", 0)
        writer.add_message(channel_id, 10, b"\0\x01\0\0", 10)
        writer.finish()


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
@unittest.skipIf(foxglove_mcap_writer is None,
                 "the Foxglove mcap package is not installed")
class McapPanelTC(unittest.TestCase):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "drive.mcap")
        write_fixture(self.path)
        self.feature = _mcap_viewer.McapPanel(mgr=self.mgr)
        self.feature.populate_menu()

    def tearDown(self):
        if self.feature.reader is not None:
            self.feature.reader.close()
        self.tmpdir.cleanup()

    def test_track_menu_carries_the_open_action(self):
        titles = [a.text() for a in self.mgr.mainWindow.menuBar().actions()]
        self.assertLess(titles.index("Profiling"), titles.index("Track"))
        self.assertLess(titles.index("Track"), titles.index("Window"))
        texts = [a.text() for a in self.mgr.menu_model.menu("Track").actions()]
        self.assertIn("Open MCAP", texts)

    def test_open_file_fills_the_dock_and_the_toggle_hides_it(self):
        reader = self.feature.open_file(self.path)
        QApplication.processEvents()
        self.assertIs(self.feature.reader, reader)
        self.assertTrue(self.feature._action.isChecked())
        self.assertFalse(self.feature._dock.isHidden())
        topics = self.feature.panel._topics
        self.assertEqual([topics.item(i).data(_mcap_viewer._TOPIC_ROLE)
                          for i in range(topics.count())], [TOPIC])
        self.assertEqual(
            self.mgr.mainWindow.dockWidgetArea(self.feature._dock),
            Qt.RightDockWidgetArea)
        second = self.feature.open_file(self.path)
        self.assertTrue(reader._file.closed)
        self.assertIs(self.feature.reader, second)
        self.feature._action.setChecked(False)
        QApplication.processEvents()
        self.assertTrue(self.feature._dock.isHidden())

        # A file the reader refuses is reported without unseating the
        # recording already open.
        broken = os.path.join(self.tmpdir.name, "broken.mcap")
        with open(broken, "wb") as fp:
            fp.write(b"not an MCAP file")
        self.feature._on_selected(broken)
        QApplication.processEvents()
        self.assertIs(self.feature.reader, second)
        self.assertEqual(self.feature.panel._error.text(), "bad magic")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
