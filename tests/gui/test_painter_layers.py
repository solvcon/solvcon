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

    def test_the_highlight_follows_a_pick_on_the_canvas(self):
        page = self.painter.panel.layers
        QtWidgets.QApplication.processEvents()
        _click(self.sub.widget(), 100, 100)
        self.assertEqual(self.widget.selectedShape, self.sid)
        page.refresh()
        selected = [row.name for row in page.shown_rows if row.selected()]
        self.assertEqual(selected, [f"Rectangle {self.sid}"])

    def test_pressing_a_row_selects_the_shape_on_the_canvas(self):
        page = self.painter.panel.layers
        QtWidgets.QApplication.processEvents()
        page.refresh()
        _click(page.shown_rows[0], 10, 10)
        self.assertEqual(self.widget.selectedShape, self.sid)
        self.assertEqual([row.name for row in page.shown_rows
                          if row.selected()], [f"Rectangle {self.sid}"])

    def test_a_pick_arms_the_select_tool_first(self):
        # The canvas draws no selection while a shape tool is armed, and
        # switching tools drops the selection, so a pick that only wrote the
        # id would leave nothing marked.
        page = self.painter.panel.layers
        self.widget.setDrawTool("circle")
        QtWidgets.QApplication.processEvents()
        page.refresh()
        _click(page.shown_rows[0], 10, 10)
        self.assertEqual(self.widget.drawTool, "select")
        self.assertEqual(self.widget.selectedShape, self.sid)
        # The rail and the Canvas menu share the action the pick triggered.
        self.assertTrue(self.painter.panel.tool_buttons["select"].isChecked())

    def test_a_pick_on_a_row_the_world_has_dropped_selects_nothing(self):
        # The list polls, so a row can outlive its shape by one tick; the pick
        # drops a dead id and heals the list without waiting for the next poll.
        page = self.painter.panel.layers
        QtWidgets.QApplication.processEvents()
        page.refresh()
        row = page.shown_rows[0]
        self.world.remove_shape(self.sid)
        _click(row, 10, 10)
        self.assertEqual(self.widget.selectedShape, -1)
        self.assertEqual(page.shown_rows, [])

    def test_a_pick_on_a_stale_row_from_another_world_selects_nothing(self):
        # Ids start over per world. A list that still names the previous
        # canvas can match a live id on the one in front; the pick drops
        # rather than selecting a shape the row never stood for.
        page = self.painter.panel.layers
        QtWidgets.QApplication.processEvents()
        page.refresh()
        row = page.shown_rows[0]
        other = self.mgr.add2DWidget()
        other_world = solvcon.WorldFp64()
        other_sid = other_world.add_circle(0, 0, 1)
        self.assertEqual(other_sid, self.sid)
        other.updateWorld(other_world)
        other.setDrawTool("select")
        other_sub = self.mgr.mdiArea.subWindowList()[-1]
        self.mgr.mdiArea.setActiveSubWindow(other_sub)
        # Leave the activation refresh on the timer; the list still names
        # the first canvas while the active one is already the second.
        _click(row, 10, 10)
        self.assertEqual(other.selectedShape, -1)
        self.assertEqual(self.widget.selectedShape, -1)
        other_sub.close()
        QtWidgets.QApplication.processEvents()

    def test_closing_the_canvas_leaves_the_list_standing(self):
        # The sub-window deletes the canvas on close, and a page that held it
        # would read freed memory on its next poll.
        page = self.painter.panel.layers
        self.sub.close()
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual(page.shown_rows, [])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
