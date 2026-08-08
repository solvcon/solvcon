# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The status readout of the oblique-shock reflection panel.

The panel is a widget, not a window, so the readout is checked here rather
than in the window lane: a `SolutionPanel` and a coarse session are all it
takes to fill the tree and read the rows back.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.apps.obsrefl import ReflectionSession
    from solvcon.pilot.apps.obsrefl._panel import SolutionPanel
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class StatusTreeTC(unittest.TestCase):
    """What `set_status` puts in which column."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def _filled(self, name='density'):
        """A panel showing one marched chunk of a coarse run."""
        panel = SolutionPanel()
        panel._field.setCurrentText(name)
        sess = ReflectionSession(nx=8, ny=3, steps_per_chunk=3)
        sess.advance()
        field = sess.field.field(name)
        panel.set_status(sess, float(field.min()), float(field.max()))
        return panel

    @staticmethod
    def _rows(panel):
        tree = panel._tree
        return [tuple(tree.topLevelItem(it).text(col) for col in range(4))
                for it in range(tree.topLevelItemCount())]

    def test_an_unstarted_panel_says_so(self):
        panel = SolutionPanel()
        self.assertEqual("not started", panel._tree.topLevelItem(0).text(0))

    def test_the_zone_rows_carry_the_computed_analytic_and_error(self):
        zones = [row for row in self._rows(self._filled())
                 if row[0].startswith("zone")]
        self.assertEqual(3, len(zones))
        # The density rises across each shock, so the analytic column has to
        # carry three ascending values rather than one repeated.
        analytic = [float(row[2]) for row in zones]
        self.assertEqual(sorted(analytic), analytic)
        self.assertEqual(3, len(set(analytic)))
        for _, computed, target, error in zones:
            # The three numbers are one comparison, so the error has to be
            # what the two columns beside it say it is.
            gap = 100.0 * (float(computed) - float(target)) / float(target)
            self.assertAlmostEqual(gap, float(error.rstrip('%')), places=1)

    def test_a_zone_with_no_analytic_value_shows_no_percent(self):
        # The transverse velocity is zero outside the wedge, and an error
        # relative to zero is not a percent of anything.
        zones = [row for row in self._rows(self._filled('vely'))
                 if row[0].startswith("zone")]
        self.assertEqual(["", ""], [zones[0][3], zones[2][3]])
        self.assertTrue(zones[1][3].endswith('%'))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
