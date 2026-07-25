# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for remembering the pilot user-interface state.
"""


import os
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.config import Config
    from solvcon.pilot.base import _gui
    from solvcon.pilot.base import _ui_state
    from PySide6 import QtGui, QtWidgets
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


class FakePart(object):
    """A UI-state part with no Qt behind it, to drive UiState alone."""

    def __init__(self, key, value=None):
        self.KEY = key
        self.value = value
        self.applied = []

    def apply(self, section):
        self.applied.append(section)
        if section is not None:
            self.value = section

    def capture(self):
        return self.value


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class GeometrySeamTC(unittest.TestCase):
    """Drive the geometry policy against a hidden, fully controlled window.

    A hidden top-level window honors resize() and move() synchronously,
    without a window manager imposing its own size, so these assertions are
    stable where a live shown window would be flaky.
    """

    def setUp(self):
        _gui.controller.build()
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1000, 600)

    def tearDown(self):
        self.win.deleteLater()
        QtWidgets.QApplication.processEvents()

    def _on_screen_origin(self):
        """A point safely inside the primary screen's available area."""
        avail = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        return avail.x() + 40, avail.y() + 40

    def test_apply_sets_size_and_location(self):
        x, y = self._on_screen_origin()
        _ui_state.apply_geometry(
            self.win, {"width": 820, "height": 540, "x": x, "y": y})
        self.assertEqual(self.win.size().width(), 820)
        self.assertEqual(self.win.size().height(), 540)
        self.assertEqual((self.win.pos().x(), self.win.pos().y()), (x, y))

    def test_apply_ignores_a_missing_section(self):
        _ui_state.apply_geometry(self.win, None)
        self.assertEqual((self.win.size().width(), self.win.size().height()),
                         (1000, 600))

    def test_apply_rejects_non_positive_size(self):
        _ui_state.apply_geometry(self.win, {"width": 0, "height": -5})
        self.assertEqual((self.win.size().width(), self.win.size().height()),
                         (1000, 600))

    def test_apply_rejects_off_screen_location(self):
        _ui_state.apply_geometry(self.win, {"x": -30000, "y": -30000})
        self.assertNotEqual((self.win.pos().x(), self.win.pos().y()),
                            (-30000, -30000))

    def test_capture_reads_the_current_geometry(self):
        x, y = self._on_screen_origin()
        self.win.resize(760, 480)
        self.win.move(x, y)
        section = _ui_state.capture_geometry(self.win)
        self.assertEqual(section["width"], 760)
        self.assertEqual(section["height"], 480)
        self.assertEqual((section["x"], section["y"]), (x, y))

    def test_window_geometry_part_round_trips_its_window(self):
        part = _ui_state.WindowGeometry(self.win)
        self.assertEqual(part.KEY, "window")
        part.apply({"width": 700, "height": 420})
        self.assertEqual(part.capture()["width"], 700)
        self.assertEqual(part.capture()["height"], 420)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class UiStateTC(unittest.TestCase):
    """UiState coordinates any number of named parts, not windows alone."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _state(self, parts):
        return _ui_state.UiState(
            mgr=self.mgr, config=Config(self.path), parts=parts)

    def test_save_groups_every_part_under_the_ui_section(self):
        self._state([FakePart("alpha", 1), FakePart("beta", 2)]).save()
        data = Config(self.path).load()
        self.assertEqual(data.get("ui"), {"alpha": 1, "beta": 2})

    def test_restore_hands_each_part_its_own_section(self):
        Config(self.path).set(
            "ui", {"alpha": "A", "beta": "B"}).save()
        alpha, beta = FakePart("alpha"), FakePart("beta")
        self._state([alpha, beta])
        self.assertEqual(alpha.applied, ["A"])
        self.assertEqual(beta.applied, ["B"])

    def test_restore_passes_none_for_an_absent_part(self):
        Config(self.path).set("ui", {"alpha": "A"}).save()
        beta = FakePart("beta", "default")
        self._state([beta])
        self.assertEqual(beta.applied, [None])
        self.assertEqual(beta.value, "default")

    def test_restore_tolerates_a_missing_ui_section(self):
        part = FakePart("alpha", "default")
        self._state([part])
        self.assertEqual(part.applied, [None])

    def test_added_part_joins_the_saved_state(self):
        state = self._state([FakePart("alpha", 1)])
        state.add(FakePart("gamma", 3))
        state.save()
        self.assertEqual(
            Config(self.path).load().get("ui"), {"alpha": 1, "gamma": 3})

    def test_window_geometry_is_the_default_part(self):
        state = _ui_state.UiState(
            mgr=self.mgr, config=Config(self.path))
        state.save()
        section = Config(self.path).load().get("ui")["window"]
        self.assertEqual(
            section, _ui_state.capture_geometry(self.mgr.mainWindow))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
