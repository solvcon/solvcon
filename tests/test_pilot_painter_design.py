# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Design page of the Painter inspector.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.canvas import _painter_design
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _click(widget, x, y):
    """Post a synthetic left-button press and release to ``widget``."""
    pos = QtCore.QPointF(x, y)
    glob = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    for etype, button, buttons in (
            (QtCore.QEvent.Type.MouseButtonPress,
             QtCore.Qt.LeftButton, QtCore.Qt.LeftButton),
            (QtCore.QEvent.Type.MouseButtonRelease,
             QtCore.Qt.LeftButton, QtCore.Qt.NoButton)):
        QtWidgets.QApplication.sendEvent(
            widget,
            QtGui.QMouseEvent(etype, pos, glob, button, buttons,
                              QtCore.Qt.NoModifier))


class _StubCanvas:
    """Stand-in for the 2D canvas, so a test can set the selection directly.

    The real canvas only changes it through a mouse gesture on a live window,
    which :class:`PainterDesignCanvasTC` covers.
    """

    def __init__(self, world):
        self.world = world
        self.selectedShape = -1
        self.repaints = 0

    def requestRepaint(self):
        self.repaints += 1


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterDesignPageTC(unittest.TestCase):
    """What the page shows for the canvas's selection."""

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.world = solvcon.WorldFp64()
        # A rectangle centered on the origin, 4 wide and 2 tall.
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.canvas = _StubCanvas(self.world)
        self.page = _painter_design.DesignPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _select(self, shape_id):
        self.canvas.selectedShape = shape_id
        self.page.refresh()

    def _header(self):
        return self.page._name.text()

    def test_nothing_selected_leaves_the_page_empty(self):
        self.assertEqual(self._header(),
                         _painter_design.DesignPage.EMPTY_TEXT)
        self.assertTrue(self.page._badge.isHidden())
        self.assertTrue(self.page._icon.isHidden())

    def test_selection_fills_the_header(self):
        self._select(self.sid)
        self.assertEqual(self._header(), f"Rectangle {self.sid}")
        self.assertFalse(self.page._badge.isHidden())
        self.assertFalse(self.page._icon.pixmap().isNull())

    def test_picking_another_shape_refreshes_the_page(self):
        # A pick moves no geometry, so a poll watching the world would miss it.
        other = self.world.add_circle(10, 10, 3)
        self._select(self.sid)
        self._select(other)
        self.assertEqual(self._header(), f"Circle {other}")

    def test_a_dead_selection_leaves_the_page_empty(self):
        # The canvas keeps the id it stored, and a query on a dead one throws.
        self._select(self.sid)
        self.world.remove_shape(self.sid)
        self.page.refresh()
        self.assertEqual(self._header(),
                         _painter_design.DesignPage.EMPTY_TEXT)

    def test_closing_the_canvas_clears_the_page(self):
        # The page asks for the canvas again on every read.
        self._select(self.sid)
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self._header(),
                         _painter_design.DesignPage.EMPTY_TEXT)

    def test_the_page_restyles_with_the_palette(self):
        # A color captured once would survive a theme switch as is.
        self._select(self.sid)
        before = self.page.styleSheet()
        icon = self.page._icon.pixmap().toImage()
        palette = QtGui.QPalette(self.page.palette())
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("red"))
        self.page.setPalette(palette)
        self.assertNotEqual(self.page.styleSheet(), before)
        self.assertNotEqual(self.page._icon.pixmap().toImage(), icon)


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class PainterDesignCanvasTC(unittest.TestCase):
    """The page against a real canvas, bound by the Painter dock."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.painter = _gui.controller.painter
        self.painter._ensure_dock()
        self.widget = self.mgr.add2DWidget()
        self.widget.setDrawTool("select")
        self.world = solvcon.WorldFp64()
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.widget.updateWorld(self.world)
        view = solvcon.ViewTransform2dFp64()
        view.pan(100.0, 100.0)
        view.zoom = 20.0
        # Set the view before showing so the resize auto-centering, which a
        # well-formed transform disables, leaves the mapping deterministic.
        self.widget.setViewTransform(view)
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        QtWidgets.QApplication.processEvents()

    def test_the_page_follows_the_active_canvas(self):
        page = self.painter.panel.design
        # Activation is delivered through a zero timer, so let it run.
        QtWidgets.QApplication.processEvents()
        _click(self.sub.widget(), 100, 100)
        self.assertEqual(self.widget.selectedShape, self.sid)
        page.refresh()
        self.assertEqual(page._name.text(), f"Rectangle {self.sid}")

    def test_closing_the_canvas_leaves_the_page_standing(self):
        # The sub-window deletes the canvas on close, and a page that held it
        # would read freed memory on its next poll.
        page = self.painter.panel.design
        self.sub.close()
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual(page._name.text(),
                         _painter_design.DesignPage.EMPTY_TEXT)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
