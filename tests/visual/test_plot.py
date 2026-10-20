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
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from PySide6.QtWidgets import QApplication
    from solvcon import pilot
    from solvcon.pilot.visual import _plot
except ImportError:
    pilot = None
    _plot = None


def _array(values):
    return solvcon.SimpleArrayFloat64(array=np.array(values, dtype='float64'))


def _wheel(widget, pos, degrees):
    """Turn the wheel by ``degrees`` with the cursor at ``pos``."""
    pos = QPointF(pos)
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(),
                        QPoint(0, round(degrees * 8)), Qt.NoButton,
                        Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def _mouse(widget, kind, pos, button, buttons):
    pos = QPointF(pos)
    event = QMouseEvent(kind, pos, widget.mapToGlobal(pos), button, buttons,
                        Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def _drag(widget, start, end):
    """Hold the left button at ``start`` and let it go at ``end``."""
    _mouse(widget, QEvent.MouseButtonPress, start, Qt.LeftButton,
           Qt.LeftButton)
    _mouse(widget, QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton)
    _mouse(widget, QEvent.MouseButtonRelease, end, Qt.LeftButton,
           Qt.NoButton)


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
        limits = widget.limits()
        low = to_screen(limits.xmin, limits.ymin)
        high = to_screen(limits.xmax, limits.ymax)
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
        self.assertLess(loose.ymin, tight.ymin)
        self.assertGreater(loose.ymax, tight.ymax)

    def test_a_log_lineplot_maps_the_exponents(self):
        widget = self._filled([1.0, 2.0, 3.0], [1e-1, 1e-3, 1e-5],
                              log_y=True)
        limits = widget.limits()
        # The limits are reported in the space that was drawn, so a decade
        # of data is a unit of ordinate.
        self.assertLess(limits.ymin, -5.0)
        self.assertGreater(limits.ymax, -1.0)

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


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class LinePlotNavigationTC(unittest.TestCase):
    """How the wheel, a drag, and a double-click move the drawn view."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.widget = _plot.LinePlotWidget()
        self.widget.add_series("curve").set_data(_array([0.0, 10.0]),
                                                 _array([0.0, 10.0]))
        self.widget.refresh()
        self.widget.resize(320, 240)
        self.rect = self.widget._lineplot_rect()

    def _spans(self):
        limits = self.widget.limits()
        return limits.xmax - limits.xmin, limits.ymax - limits.ymin

    def _under_x_ticks(self):
        return QPoint(self.rect.center().x(), self.rect.bottom() + 10)

    def _beside_y_ticks(self):
        return QPoint(self.rect.left() - 10, self.rect.center().y())

    def test_the_wheel_zooms_about_the_cursor(self):
        at = QPoint(self.rect.left() + 50, self.rect.top() + 40)
        held = self.widget._to_data(QPointF(at))
        xspan, yspan = self._spans()
        _wheel(self.widget, at, 90.0)
        # A quarter revolution zooms in by the fourth root of two, and the
        # point under the cursor stays under it.
        self.assertAlmostEqual(xspan / 2 ** 0.25, self._spans()[0])
        self.assertAlmostEqual(yspan / 2 ** 0.25, self._spans()[1])
        for expected, actual in zip(held, self.widget._to_data(QPointF(at))):
            self.assertAlmostEqual(expected, actual)
        _wheel(self.widget, at, -90.0)
        self.assertAlmostEqual(xspan, self._spans()[0])
        self.assertAlmostEqual(yspan, self._spans()[1])

    def test_the_wheel_over_the_tick_labels_zooms_one_axis(self):
        before = self.widget.limits()
        _wheel(self.widget, self._under_x_ticks(), 90.0)
        after = self.widget.limits()
        self.assertLess(after.xmax - after.xmin, before.xmax - before.xmin)
        self.assertEqual((before.ymin, before.ymax), (after.ymin, after.ymax))

        before = after
        _wheel(self.widget, self._beside_y_ticks(), 90.0)
        after = self.widget.limits()
        self.assertEqual((before.xmin, before.xmax), (after.xmin, after.xmax))
        self.assertLess(after.ymax - after.ymin, before.ymax - before.ymin)

    def test_the_wheel_over_the_title_does_nothing(self):
        before = self.widget.limits()
        _wheel(self.widget, QPoint(self.rect.center().x(), 4), 90.0)
        self.assertEqual(before, self.widget.limits())
        self.assertTrue(self.widget.model.auto_view)

    def test_a_drag_carries_the_held_point_with_the_cursor(self):
        start = QPoint(self.rect.left() + 60, self.rect.top() + 50)
        end = QPoint(self.rect.left() + 140, self.rect.top() + 90)
        held = self.widget._to_data(QPointF(start))
        _drag(self.widget, start, end)
        for expected, actual in zip(held, self.widget._to_data(QPointF(end))):
            self.assertAlmostEqual(expected, actual)

    def test_a_drag_over_the_tick_labels_pans_one_axis(self):
        before = self.widget.limits()
        start = self._under_x_ticks()
        _drag(self.widget, start, start + QPoint(40, 20))
        after = self.widget.limits()
        self.assertLess(after.xmin, before.xmin)
        self.assertEqual((before.ymin, before.ymax), (after.ymin, after.ymax))

    def test_a_navigated_view_holds_until_it_is_reset(self):
        _wheel(self.widget, self.rect.center(), 90.0)
        held = self.widget.limits()
        # More data arrives, as while a run marches; the view stays put.
        self.widget.model.series(0).set_data(_array([0.0, 20.0]),
                                             _array([0.0, 20.0]))
        self.widget.refresh()
        self.assertEqual(held, self.widget.limits())
        # A double-click follows the data again.
        center = self.rect.center()
        _mouse(self.widget, QEvent.MouseButtonDblClick, center,
               Qt.LeftButton, Qt.LeftButton)
        self.assertTrue(self.widget.model.auto_view)
        self.assertGreater(self.widget.limits().xmax, 20.0)

    def test_a_log_lineplot_navigates_the_exponents(self):
        widget = _plot.LinePlotWidget(log_y=True)
        widget.add_series("curve").set_data(_array([1.0, 2.0, 3.0]),
                                            _array([1e-1, 1e-3, 1e-5]))
        widget.refresh()
        widget.resize(320, 240)
        rect = widget._lineplot_rect()
        before = widget.limits()
        _wheel(widget, QPoint(rect.left() - 10, rect.center().y()), 90.0)
        after = widget.limits()
        self.assertLess(after.ymax - after.ymin, before.ymax - before.ymin)
        # The zoom is held through a refresh, then dropped on a reset.
        widget.refresh()
        self.assertEqual(after, widget.limits())
        widget.reset_view()
        self.assertEqual(before, widget.limits())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
