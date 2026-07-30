# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Layers page of the Painter inspector.
"""

import math
import os
import unittest

import solvcon
from pilot_ci import SKIP_PILOT_WIDGETS

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot import painter as _painter
    from solvcon.pilot.painter import _layers
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

    :class:`PainterLayersCanvasTC` covers the live canvas path, including
    selection written from outside a mouse gesture.
    """

    def __init__(self, world):
        self.world = world
        self.selectedShape = -1


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
        self.page = _layers.LayersPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _select(self, shape_id):
        self.canvas.selectedShape = shape_id
        self.page.refresh()

    def _names(self):
        return [row.name for row in self.page.shown_rows]

    def _selected(self):
        return [row.name for row in self.page.shown_rows if row.selected()]

    def test_every_object_gets_a_row_newest_first(self):
        # The world draws in registry order, so the last shape added is the
        # one on top of the canvas and the first the list names.
        self.assertEqual(self._names(),
                         [f"Circle {self.cid}", f"Rectangle {self.rid}"])
        self.assertEqual(self.page.count, "2 shapes")
        self.assertTrue(self.page._empty.isHidden())

    def test_the_metric_is_the_radius_or_the_size(self):
        self.assertEqual([row.metric for row in self.page.shown_rows],
                         ["r 3", "4 x 2"])

    def test_a_rotated_shape_reports_its_own_size(self):
        # A quarter turn leaves the rectangle 4 by 2; only the axis-aligned
        # span it covers swaps, and that is not what the row shows.
        self.world.rotate_shape(self.rid, 0.5 * math.pi, 0.0, 0.0)
        self.page.refresh()
        self.assertEqual(self.page.shown_rows[1].metric, "4 x 2")

    def test_the_highlight_follows_the_selection(self):
        # A pick moves no geometry, so a poll watching the world alone would
        # miss it.
        self.assertEqual(self._selected(), [])
        self._select(self.cid)
        self.assertEqual(self._selected(), [f"Circle {self.cid}"])
        self._select(self.rid)
        self.assertEqual(self._selected(), [f"Rectangle {self.rid}"])

    def test_the_selected_row_is_drawn_apart(self):
        self._select(self.cid)
        selected, plain = self.page.shown_rows
        self.assertNotEqual(selected.icon.toImage(), plain.icon.toImage())
        # The metric is colored by a rule reading the row's own property, and
        # a label Qt was not asked to polish again keeps the color it had.
        self.assertNotEqual(selected.metric_color, plain.metric_color)
        self._select(self.rid)
        selected, plain = self.page.shown_rows[1], self.page.shown_rows[0]
        self.assertNotEqual(selected.metric_color, plain.metric_color)

    def test_a_dead_selection_highlights_nothing(self):
        # The canvas keeps the id it stored, and a query on a dead one throws.
        self._select(self.rid)
        self.world.remove_shape(self.rid)
        self.page.refresh()
        self.assertEqual(self._selected(), [])
        self.assertEqual(self._names(), [f"Circle {self.cid}"])

    def test_a_canvas_of_the_same_boxes_still_redraws(self):
        # A rotated shape covers a wider span than it measures, so a second
        # canvas can serialize the very boxes of the first while its rows read
        # differently. The poll compares the rows, not the boxes, or the page
        # would keep showing the canvas it left.
        other = solvcon.WorldFp64()
        turned = other.add_rectangle(-1, -2, 1, 2)
        other.rotate_shape(turned, 0.5 * math.pi, 0.0, 0.0)
        other.add_circle(10, 10, 3)
        for mine, theirs in zip(self.world.shape_bbox(self.rid),
                                other.shape_bbox(turned)):
            self.assertAlmostEqual(mine, theirs)

        self.canvas = _StubCanvas(other)
        self.page.refresh()
        self.assertEqual([row.metric for row in self.page.shown_rows],
                         ["r 3", "2 x 4"])

    def test_rows_track_adding_removing_and_undo(self):
        third = self.world.add_circle(0, 0, 1)
        self.page.refresh()
        self.assertEqual(self._names()[0], f"Circle {third}")
        self.assertEqual(self.page.count, "3 shapes")

        self.world.remove_shape(self.rid)
        self.page.refresh()
        self.assertNotIn(f"Rectangle {self.rid}", self._names())
        self.assertEqual(self.page.count, "2 shapes")

        self.world.undo()
        self.page.refresh()
        self.assertIn(f"Rectangle {self.rid}", self._names())
        self.assertEqual(self.page.count, "3 shapes")

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
        self.assertEqual(self.page.count, "0 shapes")
        self.assertFalse(self.page._empty.isHidden())

    def test_closing_the_canvas_clears_the_list(self):
        # The page asks for the canvas again on every read.
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self._names(), [])
        self.assertEqual(self.page.count, "0 shapes")

    def test_controls_the_model_cannot_fill_are_greyed_out(self):
        self.assertEqual(list(self.page.placeholders),
                         ["Search", "All", "Shapes", "Guides",
                          "Add object", "Remove object"])
        for name, control in self.page.placeholders.items():
            with self.subTest(name=name):
                self.assertFalse(control.isEnabled())
                self.assertIn("needs", control.toolTip())

    def test_pressing_a_row_reports_the_shape_it_names(self):
        # The row reaches nobody itself: the press is reported, and whoever
        # owns the list decides what a pick means.
        picks = []
        self.page.picked.connect(picks.append)
        _click(self.page.shown_rows[0], 10, 10)
        self.assertEqual(picks, [self.cid])
        self.assertEqual(self.canvas.selectedShape, -1)
        _click(self.page.shown_rows[1], 10, 10)
        self.assertEqual(picks, [self.cid, self.rid])

    def test_the_page_fits_the_designed_inspector(self):
        self.assertLessEqual(self.page.sizeHint().width(),
                             _painter.PainterPanel._INSPECTOR_WIDTH)

    def test_the_page_restyles_with_the_palette(self):
        # A color captured once would survive a theme switch as is, and the
        # rows hold the icons the page handed them.
        self._select(self.cid)
        before = self.page.styleSheet()
        icon = self.page.shown_rows[0].icon.toImage()
        palette = QtGui.QPalette(self.page.palette())
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("red"))
        self.page.setPalette(palette)
        self.assertNotEqual(self.page.styleSheet(), before)
        self.assertNotEqual(self.page.shown_rows[0].icon.toImage(), icon)


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS or not solvcon.HAS_PILOT,
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
