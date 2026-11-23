# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The profile plot of the reflection figure: the exact solution as a line
and the computed values as markers, and a zoom that holds over the frames
of one field."""

import unittest

import numpy as np

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtWidgets import QApplication
        from solvcon.apps.obsrefl import _profile, _viewmodel
except ImportError:
    pilot = None


def _frame(key='density', count=101, step=1.0, wobble=0.01):
    """A frame of a jump at x = 0.5 sampled ``count`` times over the unit
    cut, the computed values ``wobble`` off the exact ones."""
    x = np.linspace(0.0, 1.0, count, dtype='float64')
    exact = np.where(x < 0.5, 1.0, 1.0 + step).astype('float64')
    computed = exact + wobble * np.sin(20.0 * x)
    return _viewmodel.ProfileFrame(
        x=x, computed=computed, exact=exact, label=key, key=key)


def _plot(case):
    plot = _profile.ProfilePlot()
    case.addCleanup(testing.dispose_widget, plot)
    plot.resize(600, 300)
    return plot


def _wheel(plot, notches):
    widget = plot.plot
    at = QPointF(widget._lineplot_rect().center())
    event = QWheelEvent(at, widget.mapToGlobal(at), QPoint(),
                        QPoint(0, int(120 * notches)), Qt.NoButton,
                        Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)


def _rgb(color):
    return (color.r, color.g, color.b)


@testing.requires_pilot
class ProfilePlotTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_the_exact_is_a_line_and_the_computed_are_markers(self):
        plot = _plot(self)
        plot.show_profile(_frame())
        self.assertEqual(101, plot.exact.size)
        self.assertEqual(101, plot.computed.size)
        self.assertEqual("density", plot.plot._ylabel)

    def test_the_series_take_the_palette_colors(self):
        plot = _plot(self)
        ink = pilot.AppInk(plot)
        want = ink.series('computed')
        self.assertEqual((want.red(), want.green(), want.blue()),
                         _rgb(plot.computed.color))
        self.assertEqual((ink.ink.red(), ink.ink.green(), ink.ink.blue()),
                         _rgb(plot.exact.color))

    def test_an_exact_that_is_not_finite_draws_the_computed_alone(self):
        plot = _plot(self)
        frame = _frame()
        plot.show_profile(frame._replace(exact=np.full(101, np.nan)))
        limits = plot.plot.limits()
        self.assertLess(limits.ymin, 1.0)
        self.assertGreater(limits.ymax, 2.0)

    def test_a_blank_draws_nothing(self):
        plot = _plot(self)
        plot.show_profile(_frame())
        plot.show_profile(None)
        self.assertIsNone(plot.frame)
        self.assertEqual(0, plot.exact.size)
        self.assertEqual(0, plot.computed.size)
        self.assertIsNone(plot.plot.limits())

    def test_the_zoom_holds_over_one_key_and_resets_on_a_new_one(self):
        plot = _plot(self)
        plot.show_profile(_frame())
        _wheel(plot, 3)
        view = plot.plot.limits()
        self.assertFalse(plot.plot.model.auto_view)
        plot.show_profile(_frame(step=1.5))
        plot.show_profile(None)
        plot.show_profile(_frame(step=1.2))
        self.assertFalse(plot.plot.model.auto_view)
        self.assertEqual(view.xmin, plot.plot.limits().xmin)
        plot.show_profile(_frame(key='pressure'))
        self.assertTrue(plot.plot.model.auto_view)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
