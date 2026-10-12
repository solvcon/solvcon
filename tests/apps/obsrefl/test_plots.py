# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The line profile of the oblique-shock reflection analysis.

The plot is a widget, not a window, so it is checked here rather than in the
window lane: an `AnalysisLinePlots` and a coarse session are all it takes to
draw a cut and navigate it.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.apps.obsrefl import ReflectionSession
    from solvcon.pilot.apps.obsrefl._plots import AnalysisLinePlots
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class AnalysisLinePlotsTC(unittest.TestCase):
    """When a navigated view of the profile holds and when it is dropped."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.sess = ReflectionSession(nx=8, ny=3, steps_per_chunk=3)
        self.sess.advance()
        self.height = self.sess.analysis.cut_height(0.5)
        self.plots = AnalysisLinePlots()
        self.profile = self.plots._profile

    def _zoomed(self, name):
        """Draw ``name`` and zoom into it."""
        self.plots.show_run(self.sess, name, self.height)
        self.profile.model.zoom_view(2.0, 2.0, 0.5, 1.0)
        self.profile.refresh()
        self.assertFalse(self.profile.model.auto_view)
        return self.profile.limits()

    def test_the_view_holds_while_the_run_marches(self):
        held = self._zoomed('density')
        self.sess.advance()
        self.plots.show_run(self.sess, 'density', self.height)
        self.assertEqual(held, self.profile.limits())

    def test_another_field_follows_its_own_range(self):
        self._zoomed('density')
        self.plots.show_run(self.sess, 'pressure', self.height)
        self.assertTrue(self.profile.model.auto_view)

    def test_a_blank_plot_drops_the_view(self):
        self._zoomed('density')
        self.plots.show_run(None, 'density', self.height)
        self.assertTrue(self.profile.model.auto_view)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
