# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The parameters of a shock-tube run and the checks that keep them valid.
"""

import math
import unittest

from solvcon.apps.shocktube import _setup


class CheckTC(unittest.TestCase):

    def test_the_defaults_pose_sod(self):
        setup = _setup.ShockTubeSetup()
        self.assertEqual({}, setup.check())
        self.assertTrue(setup.is_sod())
        self.assertEqual("Shock tube: Sod", setup.title())
        self.assertEqual("Sod, 201 points", setup.summary())

    def test_every_field_in_the_wrong_is_named_at_once(self):
        setup = _setup.ShockTubeSetup(rho_left=-1.0, gamma=math.nan,
                                      ncoord=200, time_increment="0.1")
        errors = setup.check()
        self.assertEqual(['rho_left', 'gamma', 'ncoord', 'time_increment'],
                         list(errors))
        self.assertIn("1e-6 to 1e6", errors['rho_left'])
        self.assertIn("must be odd", errors['ncoord'])
        self.assertIn("got '0.1'", errors['time_increment'])

    def test_the_tube_holds_the_diaphragm(self):
        # The solver and the exact solution burst the diaphragm at x = 0,
        # so a tube on one side of it holds one gas and no waves to judge.
        errors = _setup.ShockTubeSetup(xmin=1.0, xmax=5.0).check()
        self.assertEqual(['xmin'], list(errors))
        self.assertIn("left of the diaphragm at x = 0; got 1",
                      errors['xmin'])
        errors = _setup.ShockTubeSetup(xmin=-5.0, xmax=-1.0).check()
        self.assertEqual(['xmax'], list(errors))

    def test_a_time_step_past_the_courant_limit_names_the_limit(self):
        # Sod's fastest signal, |u| + c behind the shock, runs at 2.19, so
        # the limit on the default grid is 0.2 / 2.19.
        setup = _setup.ShockTubeSetup(time_increment=0.1)
        self.assertGreater(setup.courant(), 1.0)
        message = setup.check()['time_increment']
        self.assertIn("under 0.0912 on 201 points", message)
        # The limit read off the message is itself admissible.
        self.assertEqual({}, setup.replace(time_increment=0.0912).check())

    def test_another_problem_names_its_gases(self):
        setup = _setup.ShockTubeSetup(p_right=0.01)
        self.assertFalse(setup.is_sod())
        self.assertEqual("Shock tube: rho 1 | 0.125, p 1 | 0.01, gamma 1.4",
                         setup.title())

    def test_no_session_is_built_of_an_invalid_setup(self):
        with self.assertRaisesRegex(ValueError, "Points must be odd"):
            _setup.ShockTubeSetup(ncoord=10).session()
        sess = _setup.ShockTubeSetup(ncoord=41).session(max_steps=7,
                                                        steps_per_chunk=3)
        self.assertEqual((7, 3), (sess.max_steps, sess.steps_per_chunk))
        self.assertEqual(19, len(sess.x))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
