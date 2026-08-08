# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The run session of the oblique-shock reflection.

The stop rule is checked on residual sequences rather than on runs, so the
cases that matter (a run still falling, a plateau that never converged, a
transient dip) are all reachable without marching anything.  One coarse run
is then marched to its steady state, which is what pins the session, the
analysis, and the solver together against the analytic answer.
"""

import math
import unittest

from solvcon.pilot.apps.obsrefl import ReflectionSession
from solvcon.pilot.apps.obsrefl._session import RunHistory, SteadyDetector


class SteadyDetectorTC(unittest.TestCase):
    """The stop rule needs both of its conditions."""

    def _feed(self, residuals, **kw):
        kw.setdefault('patience', 3)
        detector = SteadyDetector(**kw)
        for residual in residuals:
            detector.update(residual)
        return detector

    def test_a_falling_residual_keeps_the_run_going(self):
        # Every chunk improves on the last, so the plateau never starts even
        # though the residual has long passed the drop.
        detector = self._feed([0.5 ** it for it in range(40)])
        self.assertFalse(detector.steady)
        self.assertEqual(0, detector.flat)

    def test_a_plateau_short_of_the_drop_keeps_the_run_going(self):
        # A run stuck at a tenth of its peak has flattened, but it has not
        # converged; only the drop condition can tell the two apart.
        detector = self._feed([1.0] + [0.1] * 20)
        self.assertFalse(detector.steady)
        self.assertGreater(detector.flat, detector.patience)

    def test_a_plateau_below_the_drop_ends_the_run(self):
        # The run has to sit flat for the whole patience: three chunks after
        # the improving one, not two.
        self.assertFalse(self._feed([1.0, 1.e-3, 1.e-3, 1.e-3]).steady)
        self.assertTrue(self._feed([1.0, 1.e-3, 1.e-3, 1.e-3, 1.e-3]).steady)

    def test_a_dip_does_not_end_the_run(self):
        # A transient dip below the drop is not a steady state; the residual
        # climbing back leaves the run above the drop again.
        detector = self._feed([1.0, 1.e-3, 1.e-3, 1.e-3, 0.5, 0.5, 0.5])
        self.assertFalse(detector.steady)


class RunHistoryTC(unittest.TestCase):

    def test_the_history_pairs_the_steps_and_drops_the_oldest(self):
        # A run marches for as long as the user lets it, so the history is
        # bounded and keeps the newest chunks.
        history = RunHistory(length=3)
        self.assertIsNone(history.last)
        for step in range(1, 6):
            history.append(step, 1.0 / step, 1.0)
        self.assertEqual(3, len(history))
        self.assertEqual([(3, 1 / 3), (4, 0.25), (5, 0.2)],
                         history.residuals)
        self.assertEqual(5, history.last.step)


class ReflectionSessionTC(unittest.TestCase):
    """The chunked march over a mesh coarse enough to keep it quick."""

    def _session(self, **kw):
        kw.setdefault('nx', 8)
        kw.setdefault('ny', 3)
        return ReflectionSession(**kw)

    def test_a_new_session_is_built_and_waiting(self):
        sess = self._session()
        self.assertEqual(0, sess.step)
        self.assertFalse(sess.finished)
        self.assertIsNone(sess.stop_reason)
        self.assertEqual(0, len(sess.history))
        self.assertTrue(math.isnan(sess.residual()))
        # The driver is built through the solver, and the analysis reads it.
        self.assertEqual(sess.shock.mesh.ncell, sess.shock.svr.ncell)
        self.assertIs(sess.field, sess.analysis.field)

    def test_advance_marches_a_chunk_and_records_it(self):
        sess = self._session(steps_per_chunk=3)
        record = sess.advance()
        self.assertEqual(3, sess.step)
        self.assertEqual(3, record.step)
        self.assertEqual(record.residual, sess.residual())
        self.assertGreater(record.residual, 0.0)
        self.assertAlmostEqual(record.mass, sess.field.calc_overall_mass())
        self.assertEqual([(3, record.residual)], sess.history.residuals)
        self.assertFalse(sess.finished)

    def test_the_last_chunk_lands_on_the_step_cap(self):
        # The cap is a step count, not a chunk count, so the final chunk is
        # trimmed instead of marching past it, and the run is over from
        # there on.
        sess = self._session(steps_per_chunk=4, max_steps=6)
        self.assertEqual(4, sess.advance().step)
        self.assertEqual(6, sess.advance().step)
        self.assertEqual('cap', sess.stop_reason)
        self.assertFalse(sess.steady)
        self.assertIsNone(sess.advance())
        self.assertEqual(6, sess.step)

    def test_stop_ends_the_run_and_keeps_what_it_measured(self):
        sess = self._session(steps_per_chunk=2)
        sess.advance()
        sess.stop()
        self.assertTrue(sess.finished)
        self.assertEqual('stopped', sess.stop_reason)
        self.assertEqual(1, len(sess.history))
        self.assertEqual(3, len(sess.zone_info()))
        # A second stop does not overwrite what ended the run.
        sess.stop()
        self.assertEqual('stopped', sess.stop_reason)


class SteadyRunTC(unittest.TestCase):
    """One run marched all the way to its steady state.

    This is the check that the reflection is solved at all: the session has
    to end itself on the steady state, and the field it settles on has to be
    the analytic three-zone answer.

    Two things keep it to a few hundredths of a second.  The mesh is coarse,
    and a coarse mesh takes a long step: the run below sits at a CFL of
    about 0.76 with a time increment twenty times the default, so it covers
    the same flow time in a twentieth of the steps.  The stop rule is also
    made less patient, since a plateau still has to fall two orders of
    magnitude from the peak to count, and waiting out twenty flat chunks
    only confirms what five already showed.

    The errors are percents on a mesh this coarse.  They shrink with
    refinement, which is what a resolution control is for, and the bounds
    below are loose enough to leave that to the panel rather than pin it.
    """

    def test_a_coarse_run_settles_onto_the_analytic_answer(self):
        sess = ReflectionSession(nx=12, ny=4, time_increment=4.e-2,
                                 steps_per_chunk=20, max_steps=2000)
        sess.detector = SteadyDetector(patience=5)
        self.assertEqual('steady', sess.run())
        self.assertLess(sess.step, sess.max_steps)
        self.assertEqual(sess.step // 20, len(sess.history))
        # The long step is only legitimate below a CFL of one.
        cflc = sess.shock.svr.cflc
        self.assertLess(float(cflc.ndarray[cflc.nghost:].max()), 1.0)
        # Every zone holds its analytic state, and the shock stands where
        # the relations put it.
        for record in sess.zone_info():
            self.assertLess(abs(record.error), 0.2)
        fit = sess.fit_incident_angle()
        self.assertLess(abs(fit.degree - fit.analytic), 10.0)
        self.assertLess(abs(sess.reflection_point().error), 0.3)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
