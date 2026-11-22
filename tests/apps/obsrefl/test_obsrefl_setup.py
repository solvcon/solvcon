# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The parameters of a reflection run say what is wrong with them: a message
names the field and the range it has to lie in, and every bad field at
once."""

import math
import unittest

from solvcon.apps.obsrefl import ReflectionSession
from solvcon.apps.obsrefl._setup import ReflectionSetup


class CheckTC(unittest.TestCase):

    def test_the_defaults_are_valid(self):
        self.assertEqual({}, ReflectionSetup().check())

    def test_each_range_bound_is_admissible(self):
        setup = ReflectionSetup(angle=1.0)
        for name, (lo, hi) in ReflectionSetup.RANGES.items():
            if 'angle' == name:
                continue
            for value in (lo, hi):
                with self.subTest(name=name, value=value):
                    errors = setup.replace(**{name: value}).check()
                    self.assertNotIn(name, errors)

    def test_every_bad_field_is_named_at_once_in_form_order(self):
        setup = ReflectionSetup(gamma=0.5, density=-1.0, nx=2, ny=1000,
                                time_increment=0.0, cell_type='hex')
        errors = setup.check()
        self.assertEqual(['gamma', 'density', 'nx', 'ny', 'time_increment',
                          'cell_type'], list(errors))
        for name, message in errors.items():
            with self.subTest(name=name):
                self.assertIn(ReflectionSetup.TITLES[name], message)
        self.assertIn("4 to 1024", errors['nx'])
        self.assertIn("1e-6 to 1", errors['time_increment'])
        self.assertIn("'hex'", errors['cell_type'])

    def test_a_count_takes_only_a_whole_number(self):
        errors = ReflectionSetup(nx=10.5, ny=8.0).check()
        self.assertEqual(['nx'], list(errors))
        self.assertIn("4 to 1024", errors['nx'])

    def test_what_is_no_number_is_named_with_the_range(self):
        errors = ReflectionSetup(density=math.nan, pressure='1',
                                 gamma=True).check()
        self.assertEqual(['gamma', 'density', 'pressure'], list(errors))
        self.assertIn("0.001 to 1e6", errors['density'])
        self.assertIn("got '1'", errors['pressure'])


class SetupTC(unittest.TestCase):

    def test_the_session_is_built_of_the_setup(self):
        setup = ReflectionSetup(mach=2.0, angle=8.0, nx=8.0, ny=3,
                                cell_type='quad')
        sess = setup.session(max_steps=40, steps_per_chunk=4)
        self.assertIsInstance(sess, ReflectionSession)
        self.assertEqual((40, 4, 0), (sess.max_steps, sess.steps_per_chunk,
                                      sess.step))
        self.assertEqual(2.0, sess.shock.mach)
        self.assertEqual(8, sess.shock.mesher.nx)

    def test_an_invalid_setup_builds_no_session(self):
        with self.assertRaisesRegex(ValueError, "1.1 to 20"):
            ReflectionSetup(mach=0.9, nx=8, ny=3).session()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
