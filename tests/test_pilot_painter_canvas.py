# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Canvas page of the Painter inspector.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import painter as _painter
    from solvcon.pilot.painter import _canvas
    from PySide6 import QtGui
except ImportError:
    pilot = None


def _view(pan_x=0.0, pan_y=0.0, zoom=1.0):
    """A view transform panned and zoomed as given."""
    view = solvcon.ViewTransform2dFp64()
    view.pan_x = pan_x
    view.pan_y = pan_y
    view.zoom = zoom
    return view


class _StubCanvas:
    """Stand-in for the 2D canvas: a view over a world.

    The real canvas hands back a detached copy of its transform, so the stub
    copies too; a page that wrote into the canvas's own transform would pass
    here and fail there.
    """

    def __init__(self, world):
        self.world = world
        self._view = _view()

    @property
    def viewTransform(self):
        return _view(self._view.pan_x, self._view.pan_y, self._view.zoom)

    def setViewTransform(self, view):
        self._view = _view(view.pan_x, view.pan_y, view.zoom)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterCanvasPageTC(unittest.TestCase):
    """What the View section reads off the canvas it is bound to."""

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.canvas = _StubCanvas(self.world)
        self.page = _canvas.CanvasPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _set_view(self, **kw):
        self.canvas.setViewTransform(_view(**kw))
        self.page.refresh()

    def test_the_readout_reads_the_zoom_as_a_percentage(self):
        self.assertEqual(self.page.zoom, "100%")
        self._set_view(zoom=0.6833)
        self.assertEqual(self.page.zoom, "68%")
        self._set_view(zoom=2.5)
        self.assertEqual(self.page.zoom, "250%")

    def test_a_far_zoom_out_keeps_the_digits_below_one(self):
        self._set_view(zoom=0.0025)
        self.assertEqual(self.page.zoom, "0.25%")

    def test_a_zoom_the_world_cannot_see_still_refreshes(self):
        # Zooming moves no geometry, so a poll watching the world alone would
        # leave the readout behind.
        self.world.add_circle(0, 0, 1)
        self.page.refresh()
        before = self.page.zoom
        self._set_view(zoom=4.0)
        self.assertNotEqual(self.page.zoom, before)
        self.assertEqual(self.page.zoom, "400%")

    def test_a_pan_leaves_the_readout_where_it_stands(self):
        # The page shows the zoom alone, and the poll compares what it shows.
        self._set_view(zoom=2.0)
        self._set_view(pan_x=120.0, pan_y=-40.0, zoom=2.0)
        self.assertEqual(self.page.zoom, "200%")

    def test_binding_another_canvas_shows_that_one(self):
        # The panel rebinds the page as the active sub-window changes, and what
        # it reads follows the canvas handed over.
        self._set_view(zoom=3.0)
        self.page.set_canvas_source(lambda: _StubCanvas(self.world))
        self.assertEqual(self.page.zoom, "100%")

    def test_without_a_canvas_the_view_section_is_dead(self):
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self.page.zoom, "")
        self.assertFalse(self.page._view.isEnabled())

    def test_sections_the_model_cannot_fill_are_greyed_out(self):
        self.assertEqual(list(self.page.placeholders),
                         ["Grid", "Axes & origin", "Background", "Units"])
        for title, section in self.page.placeholders.items():
            with self.subTest(title=title):
                self.assertFalse(section.isEnabled())
                self.assertIn("needs", section.toolTip())

    def test_the_page_fits_the_designed_inspector(self):
        self.assertLessEqual(self.page.sizeHint().width(),
                             _painter.PainterPanel._INSPECTOR_WIDTH)

    def test_the_page_restyles_with_the_palette(self):
        # A color captured once would survive a theme switch as is.
        before = self.page.styleSheet()
        palette = QtGui.QPalette(self.page.palette())
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        self.page.setPalette(palette)
        self.assertNotEqual(self.page.styleSheet(), before)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
