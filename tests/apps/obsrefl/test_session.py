# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The run session of the oblique-shock reflection.

The chunking is checked on a mesh coarse enough to march in milliseconds, and
one such run is then marched to its step cap, which is what pins the session,
the analysis, and the solver together against the analytic answer.  That
answer is the only thing a run is judged by: the march is time-accurate, so
there is nothing to be said for a step that changed the field little.
"""

import unittest

from solvcon.pilot.apps.obsrefl import ReflectionSession
from solvcon.pilot.apps.obsrefl._session import RunHistory


class RunHistoryTC(unittest.TestCase):

    def test_the_history_pairs_the_steps_and_drops_the_oldest(self):
        # A run marches for as long as the user lets it, so the history is
        # bounded and keeps the newest chunks.
        history = RunHistory(length=3)
        self.assertIsNone(history.last)
        for step in range(1, 6):
            history.append(step, 1.0 / step, 0.1 * step, 0.2 * step)
        self.assertEqual(3, len(history))
        self.assertEqual([(3, 1 / 3), (4, 0.25), (5, 0.2)], history.masses)
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
        # The driver is built through the solver, and the analysis reads it.
        self.assertEqual(sess.shock.mesh.ncell, sess.shock.svr.ncell)
        self.assertIs(sess.field, sess.analysis.field)

    def test_advance_marches_a_chunk_and_records_it(self):
        sess = self._session(steps_per_chunk=3)
        record = sess.advance()
        self.assertEqual(3, sess.step)
        self.assertEqual(3, record.step)
        self.assertAlmostEqual(record.mass, sess.field.calc_overall_mass())
        self.assertEqual([(3, record.mass)], sess.history.masses)
        # The record carries the CFL bounds of its own chunk.
        self.assertEqual(sess.field.calc_cfl_range(),
                         (record.cfl_min, record.cfl_max))
        self.assertGreater(record.cfl_min, 0.0)
        self.assertFalse(sess.finished)

    def test_the_last_chunk_lands_on_the_step_cap(self):
        # The cap is a step count, not a chunk count, so the final chunk is
        # trimmed instead of marching past it, and the run is over from
        # there on.
        sess = self._session(steps_per_chunk=4, max_steps=6)
        self.assertEqual(4, sess.advance().step)
        self.assertEqual(6, sess.advance().step)
        self.assertEqual('cap', sess.stop_reason)
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


class AnalyticAgreementTC(unittest.TestCase):
    """One run marched to its cap and judged against the analytic answer.

    This is the check that the reflection is solved at all: the field the run
    reaches has to be the analytic three-zone answer, with the shock standing
    where the relations put it.

    It runs on the default unstructured mesh, which is the one the case is
    meant to be solved on; a structured flavor would let a uniform cell size
    carry the check.  Two things keep it to a twentieth of a second.  The
    mesh is coarse, at 102 cells, and a coarse mesh takes a long step: the
    increment below is five times the default and still leaves the smallest
    cell at a CFL of 0.67.  The cap is measured rather than guessed, this
    case reaching its final field by four hundred steps and not moving when
    the cap is raised to sixteen hundred.

    The errors are percents on a mesh this coarse.  They shrink with
    refinement, which is what a resolution control is for, and the bounds
    below are loose enough to leave that to the panel rather than pin it.
    """

    def test_a_coarse_run_reaches_the_analytic_answer(self):
        sess = ReflectionSession(nx=12, ny=4, time_increment=1.e-2,
                                 steps_per_chunk=20, max_steps=400)
        self.assertEqual('cap', sess.run())
        self.assertEqual(sess.max_steps // 20, len(sess.history))
        # The long step is only legitimate below a CFL of one, and an
        # irregular mesh has no uniform cell to tune it against.
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
