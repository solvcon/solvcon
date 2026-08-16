# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The status readout of the oblique-shock reflection panel.

The panel is a widget, not a window, so the readout is checked here rather
than in the window lane: a `SolutionPanel` and a coarse session are all it
takes to fill the readout boxes and read the cells back.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.apps.obsrefl import ReflectionSession
    from solvcon.pilot.apps.obsrefl import _panel
    from solvcon.pilot.apps.obsrefl._panel import SolutionPanel
except ImportError:
    pilot = None


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class StatusReadoutTC(unittest.TestCase):
    """What `set_status` puts in which readout cell."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def _filled(self, name='density'):
        """A panel showing one marched chunk of a coarse run, with the run
        and the drawn field range it was filled from."""
        panel = SolutionPanel()
        panel._field._selector.setCurrentText(name)
        sess = ReflectionSession(nx=8, ny=3, steps_per_chunk=3)
        sess.advance()
        field = sess.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        panel.set_status(sess, vmin, vmax)
        return panel, sess, vmin, vmax

    @staticmethod
    def _zone_rows(panel):
        grid = panel._zones._grid
        rows = [tuple(grid.itemAtPosition(irow, col).widget().text()
                      for col in range(len(panel._zones.HEADERS)))
                for irow in range(1, grid.rowCount())]
        return [row for row in rows if any(row)]

    def test_an_unstarted_panel_says_so(self):
        panel = SolutionPanel()
        self.assertEqual("not started", panel._run._state.text())
        self.assertEqual("-", panel._run._progress.text())
        self.assertEqual("-", panel._field._min.text())
        self.assertEqual([], self._zone_rows(panel))

    def test_the_run_box_reads_the_march(self):
        panel, sess, _, _ = self._filled()
        self.assertEqual(f"3 / {sess.max_steps}",
                         panel._run._progress.text())
        self.assertEqual("running", panel._run._state.text())
        # One chunk is recorded, so the mass cell carries its measurement.
        self.assertEqual(_panel._number(sess.history.last.mass),
                         panel._run._mass.text())

    def test_pausing_names_the_state(self):
        panel, _, _, _ = self._filled()
        panel.set_paused(True)
        self.assertEqual("paused", panel._run._state.text())
        panel.set_paused(False)
        self.assertEqual("running", panel._run._state.text())

    def test_remesh_asks_its_owner_to_cut_the_domain_again(self):
        # The button belongs to the numerics box whose values it applies.
        panel = SolutionPanel()
        asked = []
        panel.remesh_requested = lambda: asked.append(panel.params())
        panel._numerics._nx.setValue(21)
        panel._numerics._remesh.click()
        self.assertEqual([21], [params['nx'] for params in asked])

    def test_pausing_does_not_rename_a_finished_state(self):
        # What ended a run outranks the Pause button, which the controller
        # checks when the march reaches its end.
        panel, sess, vmin, vmax = self._filled()
        sess.stop()
        panel.set_status(sess, vmin, vmax)
        panel.set_paused(True)
        self.assertEqual("stopped", panel._run._state.text())

    def test_folding_keeps_the_box_width(self):
        # Folding gives back height only; a fold that narrowed the box
        # would shift the width of the whole panel.
        panel = SolutionPanel()
        for box in panel._boxes:
            unfolded = (box.minimumSizeHint().width(), box.sizeHint().width())
            box._head.click()
            folded = (box.minimumSizeHint().width(), box.sizeHint().width())
            box._head.click()
            self.assertEqual(unfolded, folded)

    def test_every_box_folds_and_unfolds(self):
        # A folded section keeps its header and hides its content, giving
        # the room back to the boxes below it.
        panel = SolutionPanel()
        for box in panel._boxes:
            self.assertTrue(box._content.isVisibleTo(panel))
            box._head.click()
            self.assertFalse(box._content.isVisibleTo(panel))
            box._head.click()
            self.assertTrue(box._content.isVisibleTo(panel))

    def test_swapped_button_labels_keep_their_width(self):
        # The Pause and viewer buttons swap labels; the reserved minimum
        # holds the widest, so a click cannot resize the button, and with
        # it the panel.
        panel = SolutionPanel()
        run = panel._run
        for paused in (True, False):
            panel.set_paused(paused)
            self.assertLessEqual(run._pause.sizeHint().width(),
                                 run._pause.minimumWidth())
        for open_ in (True, False):
            panel.set_viewer_open(open_)
            self.assertLessEqual(run._viewer_btn.sizeHint().width(),
                                 run._viewer_btn.minimumWidth())

    def test_the_field_box_shows_the_drawn_range(self):
        panel, _, vmin, vmax = self._filled()
        self.assertEqual(_panel._number(vmin), panel._field._min.text())
        self.assertEqual(_panel._number(vmax), panel._field._max.text())

    def test_the_zone_rows_carry_the_computed_analytic_and_error(self):
        panel, _, _, _ = self._filled()
        zones = self._zone_rows(panel)
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
        panel, _, _, _ = self._filled('vely')
        zones = self._zone_rows(panel)
        self.assertEqual(["", ""], [zones[0][3], zones[2][3]])
        self.assertTrue(zones[1][3].endswith('%'))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
