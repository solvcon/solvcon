# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the widget that paints a plot model.

The plot is a widget, not a window, so it is checked here rather than in
the window lane: a manager for the QApplication and a filled model are all
it takes to map the axes and render the curves.
"""

import math
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.visual import _plot
except ImportError:
    pilot = None
    _plot = None


def _array(values):
    return solvcon.SimpleArrayFloat64(array=np.array(values, dtype='float64'))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class LinePlotTickTC(unittest.TestCase):
    """The tick placement the axes are labelled from."""

    def test_linear_ticks_are_round_and_cover_the_span(self):
        ticks = _plot._nice_ticks(0.0, 10.0, want=5)
        self.assertEqual([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], ticks)
        # Every tick has to land inside the span it was asked for, or the
        # axis is labelled outside its own frame.
        for lo, hi in ((0.3, 0.7), (-5.0, 5.0), (1e4, 1.2e4)):
            for tick in _plot._nice_ticks(lo, hi):
                self.assertGreaterEqual(tick, lo)
                self.assertLessEqual(tick, hi)

    def test_a_span_of_nothing_has_no_ticks(self):
        # A flat curve leaves a zero span; ticking it would divide by it.
        self.assertEqual([], _plot._nice_ticks(1.0, 1.0))
        self.assertEqual([], _plot._nice_ticks(1.0, float('nan')))

    def test_ticks_are_counted_and_not_accumulated(self):
        # Where the span is small beside the offset, adding the step rounds
        # back to where it started and a walk along the axis never ends.
        # This runs inside paintEvent, so it takes the GUI thread with it.
        ticks = _plot._nice_ticks(1e16, 1e16 + 4.0)
        self.assertGreater(len(ticks), 0)
        self.assertLessEqual(len(ticks), 12)

    def test_a_log_tick_off_a_whole_decade_reads_its_own_value(self):
        # A range inside one decade is ticked at its own ends, which are
        # not powers of ten.  Labelling those as powers of ten puts the
        # axis off by a factor the reader has no way to see.
        self.assertEqual(["1e-4", "1e-3"],
                         [_plot._decade_label(it) for it in (-4.0, -3.0)])
        self.assertEqual(["1.91", "5.235"],
                         [_plot._decade_label(it)
                          for it in _plot._decade_ticks(0.2811, 0.7189)])

    def test_log_ticks_are_whole_decades(self):
        self.assertEqual([-4.0, -3.0, -2.0], _plot._decade_ticks(-4.2, -1.8))
        # A range inside one decade still gets its ends marked, so the axis
        # is never left blank.
        self.assertEqual([-2.4, -2.1], _plot._decade_ticks(-2.4, -2.1))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class LinePlotWidgetTC(unittest.TestCase):
    """What the widget maps and what it draws."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def _filled(self, xs, ys, **kw):
        widget = _plot.LinePlotWidget(**kw)
        series = widget.add_series("curve")
        series.set_data(_array(xs), _array(ys))
        widget.refresh()
        return widget

    def test_a_lineplot_with_no_data_draws_nothing_and_survives(self):
        widget = _plot.LinePlotWidget(title="empty")
        widget.refresh()
        self.assertIsNone(widget.limits())
        widget.resize(300, 200)
        self.assertFalse(widget.grab().isNull())

    def test_the_axes_are_scaled_apart(self):
        # A step count against a small quantity share no scale, so the
        # model's own view, which carries one zoom, would flatten one of
        # them to nothing.  Each axis is stretched onto the frame here.
        widget = self._filled([0.0, 1000.0], [0.0, 1e-3])
        widget.resize(320, 240)
        rect = widget._lineplot_rect()
        to_screen = widget._mapper(rect)
        xmin, xmax, ymin, ymax = widget.limits()
        low = to_screen(xmin, ymin)
        high = to_screen(xmax, ymax)
        self.assertAlmostEqual(rect.left(), low.x())
        self.assertAlmostEqual(rect.bottom(), low.y())
        self.assertAlmostEqual(rect.left() + rect.width(), high.x())
        self.assertAlmostEqual(rect.bottom() - rect.height(), high.y())

    def test_the_model_scales_the_lineplot(self):
        # The box drawn in is the model's autoscale rather than a second
        # rule here, so the margin the model carries is what opens it.
        widget = self._filled([0.0, 1.0, 2.0], [5.0, 5.0, 5.0])
        self.assertEqual(widget.model.view_limits(), widget.limits())
        tight = widget.limits()
        widget.model.margin = 0.5
        widget.refresh()
        loose = widget.limits()
        self.assertLess(loose[2], tight[2])
        self.assertGreater(loose[3], tight[3])

    def test_a_log_lineplot_maps_the_exponents(self):
        widget = self._filled([1.0, 2.0, 3.0], [1e-1, 1e-3, 1e-5],
                              log_y=True)
        _, _, ymin, ymax = widget.limits()
        # The limits are reported in the space that was drawn, so a decade
        # of data is a unit of ordinate.
        self.assertLess(ymin, -5.0)
        self.assertGreater(ymax, -1.0)

    def test_a_log_lineplot_drops_what_it_cannot_place(self):
        # Zero and negative values have no logarithm; keeping them would
        # break the whole curve rather than leave a gap in it.
        widget = self._filled([1.0, 2.0, 3.0, 4.0],
                              [1e-2, 0.0, -1.0, 1e-4], log_y=True)
        drawn = widget._points(widget.model.series(0))
        self.assertEqual([1.0, 4.0], [x for x, _ in drawn])
        self.assertTrue(all(math.isfinite(y) for _, y in drawn))

    def test_the_curve_is_painted_between_the_axes(self):
        # The widget draws rather than lays out, so only rendering says
        # whether it works.  A rising line has to leave ink off the
        # frame's own edges.
        widget = self._filled([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        widget.resize(320, 240)
        pixmap = widget.grab()
        image = pixmap.toImage()
        self.assertFalse(image.isNull())
        scale = pixmap.devicePixelRatio()
        rect = widget._lineplot_rect()
        background = image.pixelColor(round(rect.left() * scale) + 4,
                                      round(rect.top() * scale) + 4)
        inked = 0
        for step in range(1, 20):
            at = step / 20.0
            point = widget._mapper(rect)(2.0 * at, 2.0 * at)
            color = image.pixelColor(round(point.x() * scale),
                                     round(point.y() * scale))
            inked += int(color != background)
        self.assertGreater(inked, 10)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
