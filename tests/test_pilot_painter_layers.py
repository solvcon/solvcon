# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Layers page of the Painter inspector.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.canvas import _painter_gui, _painter_layers
    from PySide6 import QtGui, QtWidgets
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


class _StubCanvas:
    """Stand-in for the 2D canvas, so a test can hand the page a world.

    The real canvas reaches the page through the manager, which
    :class:`PainterLayersCanvasTC` covers.
    """

    def __init__(self, world):
        self.world = world


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterLayersPageTC(unittest.TestCase):
    """What the list shows for the objects a world holds."""

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.rid = self.world.add_rectangle(-2, -1, 2, 1)
        self.cid = self.world.add_circle(10, 10, 3)
        self.canvas = _StubCanvas(self.world)
        self.page = _painter_layers.LayersPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _names(self):
        return [row.name for row in self.page.shown_rows]

    def test_every_object_gets_a_row_newest_first(self):
        # The world draws in registry order, so the last shape added is the
        # one on top of the canvas and the first the list names.
        self.assertEqual(self._names(),
                         [f"Circle {self.cid}", f"Rectangle {self.rid}"])
        self.assertTrue(self.page._empty.isHidden())

    def test_rows_track_adding_removing_and_undo(self):
        third = self.world.add_circle(0, 0, 1)
        self.page.refresh()
        self.assertEqual(self._names()[0], f"Circle {third}")

        self.world.remove_shape(self.rid)
        self.page.refresh()
        self.assertNotIn(f"Rectangle {self.rid}", self._names())

        self.world.undo()
        self.page.refresh()
        self.assertIn(f"Rectangle {self.rid}", self._names())

    def test_an_object_with_no_icon_still_gets_a_row(self):
        # The icon set draws the tools; the world holds shapes beyond them.
        line = self.world.add_polyline([(0, 0), (1, 1), (2, 0)])
        self.page.refresh()
        self.assertEqual(self._names()[0], f"Polyline {line}")
        self.assertTrue(self.page.shown_rows[0].icon.isNull())

    def test_an_empty_world_shows_the_empty_note(self):
        self.canvas = _StubCanvas(solvcon.WorldFp64())
        self.page.refresh()
        self.assertEqual(self._names(), [])
        self.assertFalse(self.page._empty.isHidden())

    def test_closing_the_canvas_clears_the_list(self):
        # The page asks for the canvas again on every read.
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self._names(), [])

    def test_the_page_fits_the_designed_inspector(self):
        self.assertLessEqual(self.page.sizeHint().width(),
                             _painter_gui.PainterPanel._INSPECTOR_WIDTH)

    def test_the_page_restyles_with_the_palette(self):
        # A color captured once would survive a theme switch as is, and the
        # rows hold the icons the page handed them.
        before = self.page.styleSheet()
        icon = self.page.shown_rows[0].icon.toImage()
        palette = QtGui.QPalette(self.page.palette())
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        self.page.setPalette(palette)
        self.assertNotEqual(self.page.styleSheet(), before)
        self.assertNotEqual(self.page.shown_rows[0].icon.toImage(), icon)


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class PainterLayersCanvasTC(unittest.TestCase):
    """The page against a real canvas, bound by the Painter dock."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.painter = _gui.controller.painter
        self.painter._ensure_dock()
        # The manager is shared with every other pilot suite, and a canvas one
        # of them left open would answer for this one's once it closes.
        for subwin in self.mgr.mdiArea.subWindowList():
            subwin.close()
        QtWidgets.QApplication.processEvents()
        self.widget = self.mgr.add2DWidget()
        self.world = solvcon.WorldFp64()
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.widget.updateWorld(self.world)
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        QtWidgets.QApplication.processEvents()

    def tearDown(self):
        # The manager and its Painter dock outlive this class, so a canvas
        # left open would answer for the tests that follow.
        self.sub.close()
        QtWidgets.QApplication.processEvents()

    def test_the_list_follows_the_active_canvas(self):
        page = self.painter.panel.layers
        # Activation is delivered through a zero timer, so let it run.
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual([row.name for row in page.shown_rows],
                         [f"Rectangle {self.sid}"])

    def test_closing_the_canvas_leaves_the_list_standing(self):
        # The sub-window deletes the canvas on close, and a page that held it
        # would read freed memory on its next poll.
        page = self.painter.panel.layers
        self.sub.close()
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual(page.shown_rows, [])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
