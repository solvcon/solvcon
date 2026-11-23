# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the presenter reads of a shock-tube run for the widgets: the
verdict of the card, the frames of the figure, the state of the strip, and
the summaries of the groups.  None of it needs Qt.
"""

import unittest

import numpy as np

from solvcon.apps.shocktube import _presenter, _setup, _viewmodel
from solvcon.pilot import panel


def _session(steps=50, **kw):
    sess = _setup.ShockTubeSetup(**kw).session(max_steps=steps,
                                               steps_per_chunk=10)
    sess.run()
    return sess


class ResultsTC(unittest.TestCase):

    def test_the_card_judges_the_star_state_and_the_fronts(self):
        results = _presenter.results(_session())
        self.assertEqual(['p*', 'u*', 'rho* left', 'rho* right'],
                         [row.label for row in results.states])
        self.assertEqual('good', results.states[0].error.tone)
        self.assertEqual("0.3031", results.states[0].exact.text)
        self.assertEqual(['contact x', 'shock x'],
                         [row.label for row in results.fronts])
        self.assertEqual(['mass drift', 'L1 density', 'CFL'],
                         [label for label, _ in results.rows])
        self.assertEqual("0.55", dict(results.rows)['CFL'])
        self.assertIsNone(results.note)

    def test_a_fresh_run_has_nothing_to_judge_yet(self):
        results = _presenter.results(_session(0))
        self.assertEqual({"no cells"},
                         {row.computed.text for row in results.states})
        self.assertEqual({"not formed"},
                         {row.computed.text for row in results.fronts})
        self.assertEqual("-", dict(results.rows)['CFL'])

    def test_a_wave_out_of_the_tube_voids_the_comparison(self):
        results = _presenter.results(_session(50, xmin=-2.0, xmax=2.0,
                                              ncoord=41,
                                              time_increment=0.05))
        self.assertEqual('bad', results.note[1])


class FrameTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = _session(20)

    def test_the_profile_draws_what_the_display_asks(self):
        display = _viewmodel.Display(quantities=('pressure', 'entropy'))
        frame = _presenter.profile_frame(self.sess, display)
        self.assertEqual(('pressure', 'entropy'), frame.quantities)
        self.assertEqual(_presenter.EXACT_POINTS, len(frame.exact[0]))
        self.assertEqual((-10.0, 10.0), frame.xlim)
        self.assertEqual("Sod, step 20, t = 1, waves R-C-S", frame.caption)
        self.assertEqual(['head', 'tail', 'contact', 'shock'],
                         [label for label, _ in frame.fronts])
        cells = _presenter.profile_frame(self.sess,
                                         display._replace(exact='cells'))
        self.assertEqual(2 * len(self.sess.x), len(cells.exact[0]))
        self.assertEqual("exact cell means", cells.legend[0][1])

    def test_the_xt_diagram_has_a_row_per_record(self):
        frame = _presenter.xt_frame(self.sess)
        self.assertEqual((3, len(self.sess.x)), frame.rows.shape)
        np.testing.assert_allclose([0.0, 0.5, 1.0], frame.times)
        self.assertEqual(1.0, frame.t_end)
        self.assertEqual(['rarefaction', 'contact', 'shock'],
                         [kind for kind, _ in frame.waves])

    def test_the_strip_runs_on_the_time_increment(self):
        self.assertEqual(panel.StripState('paused', 20, 20, 1.0, 1.0, '',
                                          True, False),
                         _presenter.strip_state(self.sess, 'paused',
                                                dirty=True, can_reset=False))
        self.assertEqual(0, _presenter.strip_state(None, 'ready').max_steps)


class SummaryTC(unittest.TestCase):

    def test_the_summaries_mark_what_waits(self):
        setup = _setup.ShockTubeSetup()
        self.assertEqual("p 1 | 0.1, rho 1 | 0.125",
                         _presenter.problem_summary(setup, False))
        self.assertEqual("201 on [-10, 10], dt 0.05  (pending)",
                         _presenter.numerics_summary(setup, True))
        self.assertTrue(_presenter.numerics_summary(
            setup, True, True).endswith("(invalid)"))
        self.assertEqual("3 panels, fronts, x-t",
                         _presenter.display_summary(_viewmodel.Display()))
        self.assertEqual("entropy, cell means", _presenter.display_summary(
            _viewmodel.Display(quantities=('entropy',), exact='cells',
                               fronts=False, xt=False)))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
