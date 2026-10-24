# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The range the oblique-shock reflection pins its legend to."""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.apps.obsrefl import ReflectionSession
    from solvcon.pilot.visual import ColorBar
except ImportError:
    pilot = None


@testing.requires_pilot
class ReflectionLegendTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def _scaled(self, name):
        """A bar filled from one marched chunk of a coarse run, with the
        run and the drawn field range it was filled from."""
        sess = ReflectionSession(nx=8, ny=3, steps_per_chunk=3)
        sess.advance()
        field = sess.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        bar = ColorBar()
        bar.show_scale(*sess.analysis.color_range(name, vmin, vmax))
        return bar, sess, vmin, vmax

    def test_the_legend_spans_the_range_the_viewer_pins(self):
        # The legend is what says which color means what, so it has to be
        # drawn over the range the viewer colors against, not the frame's
        # own: a field short of the answer would otherwise read as if it
        # had arrived.
        bar, sess, vmin, vmax = self._scaled('density')
        self.assertEqual(sess.analysis.color_range('density', vmin, vmax),
                         (bar.lo, bar.hi))
        self.assertLessEqual(bar.lo, vmin)
        self.assertGreaterEqual(bar.hi, vmax)

    def test_a_field_with_no_analytic_value_spans_the_frame(self):
        # The CFL number belongs to the discretization, not to the flow, so
        # there is no analytic value to widen its range to and the ramp is
        # the frame's own.
        bar, _, vmin, vmax = self._scaled('cfl')
        self.assertEqual((vmin, vmax), (bar.lo, bar.hi))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
