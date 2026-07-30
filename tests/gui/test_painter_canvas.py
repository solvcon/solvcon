# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Canvas page of the Painter inspector, against a real
canvas.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from PySide6 import QtWidgets
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _view(pan_x=0.0, pan_y=0.0, zoom=1.0):
    """A view transform panned and zoomed as given."""
    view = solvcon.ViewTransform2dFp64()
    view.pan_x = pan_x
    view.pan_y = pan_y
    view.zoom = zoom
    return view


def _assert_shows(case, canvas, *points):
    """Fail unless every world point given lands inside ``canvas``."""
    width, height = canvas.viewportSize
    for point in points:
        x, y = canvas.viewTransform.screen_from_world(*point)
        case.assertTrue(0 < x < width, x)
        case.assertTrue(0 < y < height, y)


def _assert_centered(case, canvas, point):
    """Fail unless the world ``point`` lands in the middle of ``canvas``."""
    width, height = canvas.viewportSize
    x, y = canvas.viewTransform.screen_from_world(*point)
    case.assertAlmostEqual(x, 0.5 * width, places=3)
    case.assertAlmostEqual(y, 0.5 * height, places=3)


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class PainterCanvasCanvasTC(unittest.TestCase):
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
        self.world.add_rectangle(-2, -1, 2, 1)
        self.widget.updateWorld(self.world)
        # Set the view before showing so the resize auto-centering, which a
        # well-formed transform disables, leaves the mapping deterministic.
        self.widget.setViewTransform(_view(100.0, 100.0, 20.0))
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        QtWidgets.QApplication.processEvents()

    def tearDown(self):
        self.sub.close()
        QtWidgets.QApplication.processEvents()

    def _page(self):
        page = self.painter.panel.canvas
        # Activation is delivered through a zero timer, so let it run.
        QtWidgets.QApplication.processEvents()
        page.refresh()
        return page

    def test_the_readout_follows_the_active_canvas(self):
        self.assertEqual(self._page().zoom, "2000%")

    def test_the_canvas_reports_the_space_the_view_maps_into(self):
        self.assertEqual(
            self.widget.viewportSize,
            (self.sub.widget().width(), self.sub.widget().height()))

    def test_fit_frames_the_shape_on_the_canvas(self):
        self._page().buttons["Fit"].click()
        _assert_shows(self, self.widget, (-2, -1), (2, 1))

    def test_fit_centers_a_world_too_wide_for_the_zoom_band(self):
        # Pan must use the clamped zoom, not the requested one.
        self.world.add_point(-1e12, -1e12, 0)
        self.world.add_point(1e12, 1e12, 0)
        self._page().buttons["Fit"].click()
        _assert_centered(self, self.widget, (0.0, 0.0))

    def test_closing_the_canvas_leaves_the_page_standing(self):
        page = self._page()
        self.sub.close()
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual(page.zoom, "")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
