# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Canvas page of the Painter inspector.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot import painter as _painter
    from solvcon.pilot.painter import _canvas
    from PySide6 import QtGui, QtWidgets
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


class _StubCanvas:
    """Stand-in for the 2D canvas.

    The real canvas hands back a detached copy of its transform and takes a
    whole one back, so the stub copies in both directions; a page that wrote
    into the canvas's own transform would pass here and fail there.

    :class:`PainterCanvasCanvasTC` covers the live canvas path, including the
    zoom clamp the stub leaves alone.
    """

    SIZE = (400, 300)

    def __init__(self, world):
        self.world = world
        self.viewportSize = self.SIZE
        self.repaints = 0
        self._view = _view()

    @property
    def viewTransform(self):
        return _view(self._view.pan_x, self._view.pan_y, self._view.zoom)

    def setViewTransform(self, view):
        self._view = _view(view.pan_x, view.pan_y, view.zoom)

    def requestRepaint(self):
        self.repaints += 1


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterCanvasPageTC(unittest.TestCase):
    """What the View section reads and what its buttons do to the view."""

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.world = solvcon.WorldFp64()
        self.canvas = _StubCanvas(self.world)
        self.page = _canvas.CanvasPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _shown_center(self):
        """The world point the middle of the canvas shows."""
        width, height = self.canvas.viewportSize
        return self.canvas.viewTransform.world_from_screen(
            0.5 * width, 0.5 * height)

    def _assert_shows(self, *points):
        _assert_shows(self, self.canvas, *points)

    def _fit(self):
        """Press Fit, after the poll has seen what the world now holds.

        Qt drops a press on a disabled button, and Fit is disabled until the
        page has read a world with something in it.
        """
        self.page.refresh()
        self.page.buttons["Fit"].click()

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
        self._set_view(zoom=2.0)
        self._set_view(pan_x=120.0, pan_y=-40.0, zoom=2.0)
        self.assertEqual(self.page.zoom, "200%")

    def test_binding_another_canvas_shows_that_one(self):
        self.world.add_circle(0, 0, 1)
        self._set_view(zoom=3.0)
        self.assertTrue(self.page.buttons["Fit"].isEnabled())
        self.page.set_canvas_source(lambda: _StubCanvas(solvcon.WorldFp64()))
        self.assertEqual(self.page.zoom, "100%")
        self.assertFalse(self.page.buttons["Fit"].isEnabled())

    def test_the_buttons_stand_in_design_order(self):
        self.assertEqual(list(self.page.buttons), ["Fit", "100%", "Center"])

    def test_a_hundred_percent_keeps_the_view_center(self):
        self._set_view(pan_x=30.0, pan_y=-70.0, zoom=8.0)
        center = self._shown_center()
        self.page.buttons["100%"].click()
        self.assertEqual(self.canvas.viewTransform.zoom, 1.0)
        for got, want in zip(self._shown_center(), center):
            self.assertAlmostEqual(got, want)
        self.assertEqual(self.page.zoom, "100%")
        self.assertEqual(self.canvas.repaints, 1)

    def test_center_preserves_the_zoom(self):
        self.world.add_circle(10, 20, 3)
        self._set_view(pan_x=5.0, pan_y=5.0, zoom=3.0)
        self.page.buttons["Center"].click()
        self.assertEqual(self.canvas.viewTransform.zoom, 3.0)
        for got, want in zip(self._shown_center(), (10.0, 20.0)):
            self.assertAlmostEqual(got, want)
        self.assertEqual(self.canvas.repaints, 1)

    def test_center_on_an_empty_world_shows_the_origin(self):
        self._set_view(pan_x=400.0, pan_y=-90.0, zoom=2.0)
        self.page.buttons["Center"].click()
        self.assertEqual(self.canvas.viewTransform.zoom, 2.0)
        for got in self._shown_center():
            self.assertAlmostEqual(got, 0.0)

    def test_fit_frames_the_content_inside_the_canvas(self):
        self.world.add_rectangle(-20, -10, 20, 10)
        self._fit()
        self.assertAlmostEqual(self.canvas.viewTransform.zoom,
                               _canvas.CanvasPage._FIT_MARGIN
                               * self.canvas.viewportSize[0] / 40)
        self._assert_shows((-20, -10), (20, 10))
        for got in self._shown_center():
            self.assertAlmostEqual(got, 0.0)

    def test_fit_frames_a_shape_where_the_poll_never_saw_it(self):
        # Counts do not move on translate; Fit must re-measure.
        sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.page.refresh()
        before = self.page._key
        self.world.translate_shape(sid, 500.0, 500.0)
        self.assertEqual(self.page._key, before)
        self.page.buttons["Fit"].click()
        self._assert_shows((498, 499), (502, 501))

    def test_fit_covers_the_geometry_no_shape_owns(self):
        self.world.add_segment(solvcon.Point3dFp64(100, 100, 0),
                               solvcon.Point3dFp64(140, 130, 0))
        self.world.add_point(80, 90, 0)
        self._fit()
        self._assert_shows((80, 90), (140, 130))

    def test_fit_fills_the_canvas_with_content_smaller_than_a_unit(self):
        # Must not floor the span to _MIN_SPAN.
        self.world.add_rectangle(-0.1, -0.1, 0.1, 0.1)
        self._fit()
        self.assertAlmostEqual(self.canvas.viewTransform.zoom,
                               _canvas.CanvasPage._FIT_MARGIN
                               * self.canvas.viewportSize[1] / 0.2)

    def test_fit_frames_a_segment_flat_on_one_axis_by_the_other(self):
        self.world.add_segment(solvcon.Point3dFp64(0, 5, 0),
                               solvcon.Point3dFp64(0.1, 5, 0))
        self._fit()
        self.assertAlmostEqual(self.canvas.viewTransform.zoom,
                               _canvas.CanvasPage._FIT_MARGIN
                               * self.canvas.viewportSize[0] / 0.1)
        self._assert_shows((0, 5), (0.1, 5))

    def test_fit_centers_bounds_whose_sum_would_overflow(self):
        self.world.add_point(9e307, 9e307, 0)
        self.world.add_point(1e308, 1e308, 0)
        self._fit()
        # Written out rather than figured the way the page figures it, so the
        # test does not agree with a page that halves the wrong way.
        middle = 9.5e307
        _assert_centered(self, self.canvas, (middle, middle))

    def test_fit_gives_a_world_of_no_extent_a_span(self):
        self.world.add_point(4, -6, 0)
        self._fit()
        zoom = self.canvas.viewTransform.zoom
        self.assertGreater(zoom, 0.0)
        self.assertLess(zoom, float("inf"))
        for got, want in zip(self._shown_center(), (4.0, -6.0)):
            self.assertAlmostEqual(got, want)

    def test_fit_is_greyed_out_while_there_is_nothing_to_fit(self):
        self.assertFalse(self.page.buttons["Fit"].isEnabled())
        for name in ("100%", "Center"):
            self.assertTrue(self.page.buttons[name].isEnabled())
        self.world.add_circle(0, 0, 1)
        self.page.refresh()
        self.assertTrue(self.page.buttons["Fit"].isEnabled())

    def test_fit_on_a_removed_shape_is_a_no_op_until_undo(self):
        # Pad leftovers keep counts non-zero, so Fit stays enabled and no-ops
        # until undo brings drawable geometry back.
        sid = self.world.add_circle(0, 0, 1)
        self.page.refresh()
        self.world.remove_shape(sid)
        self.page.refresh()
        self.assertTrue(self.page.buttons["Fit"].isEnabled())
        before = self.canvas.viewTransform
        self.page.buttons["Fit"].click()
        after = self.canvas.viewTransform
        self.assertEqual((after.pan_x, after.pan_y, after.zoom),
                         (before.pan_x, before.pan_y, before.zoom))
        self.assertEqual(self.canvas.repaints, 0)
        self.world.undo()
        self._fit()
        self._assert_shows((-1, -1), (1, 1))

    def test_a_center_the_pan_cannot_reach_leaves_the_view_alone(self):
        # Overflowing pan must not leave a half-applied zoom.
        self.world.add_point(1e307, 1e307, 0)
        self.page.refresh()
        before = self.canvas.viewTransform
        self.page.buttons["Fit"].click()
        after = self.canvas.viewTransform
        self.assertEqual((after.pan_x, after.pan_y, after.zoom),
                         (before.pan_x, before.pan_y, before.zoom))
        self.assertEqual(self.canvas.repaints, 0)

    def test_without_a_canvas_the_view_section_is_dead(self):
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self.page.zoom, "")
        for name, button in self.page.buttons.items():
            with self.subTest(name=name):
                self.assertFalse(button.isEnabled())

    def test_a_press_after_the_canvas_is_gone_reaches_nothing(self):
        self.world.add_circle(0, 0, 1)
        self._set_view(pan_x=10.0, pan_y=-20.0, zoom=2.0)
        stub = self.canvas
        before = stub.viewTransform
        self.canvas = None
        for name in self.page.buttons:
            with self.subTest(name=name):
                self.page.buttons[name].click()
                self.assertEqual(
                    (stub.viewTransform.pan_x, stub.viewTransform.pan_y,
                     stub.viewTransform.zoom),
                    (before.pan_x, before.pan_y, before.zoom))
        self.page.refresh()
        self.assertEqual(self.page.zoom, "")

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
