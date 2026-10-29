# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The config column of the shock-tube app, built without a window."""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.shocktube import _panel
except ImportError:
    pilot = None


SOD = {'gamma': 1.4, 'p_left': 1.0, 'rho_left': 1.0, 'p_right': 0.1,
       'rho_right': 0.125, 'xmin': -10, 'xmax': 10, 'ncoord': 201,
       'time_increment': 0.05, 'time_interval': 10, 'max_steps': 50}


@testing.requires_pilot
class ConfigColumnTC(unittest.TestCase):

    def setUp(self):
        self.column = _panel.ConfigColumn()
        self.column.show_values(SOD)
        self.applied = []
        self.column.applied = self.applied.append

    def _type(self, key, text):
        field = self.column.fields[key]
        field.setText(text)
        field.textEdited.emit(text)

    def test_the_fields_read_back_the_values_shown(self):
        self.assertEqual((SOD, {}), self.column.read())
        self.assertFalse(self.column.is_dirty())
        # The solver takes the counts as whole numbers.
        self.assertIsInstance(self.column.read()[0]['ncoord'], int)
        self.assertEqual(["p 1 | 0.1, rho 1 | 0.125", "201 on [-10, 10]",
                          "50 x 0.05"],
                         [group.summary() for group in self.column.groups])
        # A value with more digits than a short form keeps reads back
        # untouched, or the column would ask for an Apply at once.
        self.column.show_values(dict(SOD, gamma=0.1 + 1.3))
        self.assertFalse(self.column.is_dirty())

    def test_one_group_opens_at_a_time(self):
        problem, grid, march = self.column.groups
        self.assertEqual([True, False, False],
                         [group.is_open for group in self.column.groups])
        grid._head.click()
        self.assertEqual([False, True, False],
                         [group.is_open for group in self.column.groups])
        grid._head.click()
        self.assertEqual([False, False, False],
                         [group.is_open for group in self.column.groups])

    def test_an_edit_waits_for_apply(self):
        self._type('max_steps', "20")
        self.assertTrue(self.column.is_dirty())
        self.assertEqual([], self.applied)
        # The folded head already says what Apply will build.
        self.assertEqual("20 x 0.05", self.column.groups[2].summary())
        self.column.apply_button.click()
        self.assertEqual(dict(SOD, max_steps=20), self.applied[-1])
        self.assertFalse(self.column.is_dirty())

    def test_a_value_the_solver_cannot_take_holds_apply(self):
        # An even count would raise in the solver, after the old run was
        # dropped.
        self._type('ncoord', "200")
        self.assertEqual(['ncoord'], self.column.invalid_fields())
        self._type('gamma', "x")
        # Only once every field holds a number do the ranges show.
        self.assertEqual(['gamma'], self.column.invalid_fields())
        self.assertEqual("Gamma is not a number.", self.column.error_text())
        self.assertFalse(self.column.apply_button.isEnabled())
        self._type('gamma', "1.4")
        self.assertEqual("Points must be odd and at least 5.",
                         self.column.error_text())
        self.column._on_apply()
        self.assertEqual([], self.applied)
        self._type('ncoord', "101")
        self.assertEqual([], self.column.invalid_fields())
        self.assertTrue(self.column.apply_button.isEnabled())
        # A rejected edit keeps the head on the applied values.
        self._type('xmax', "-20")
        self.assertEqual(["xmax"], self.column.invalid_fields())
        self.assertEqual("201 on [-10, 10]", self.column.groups[1].summary())

    def test_check_names_each_value_out_of_range(self):
        bad = dict(SOD, gamma=1.0, p_left=0.0, rho_right=-1.0, xmax=-10,
                   ncoord=3, time_increment=0.0, max_steps=0,
                   time_interval=0)
        self.assertEqual(
            {'gamma', 'p_left', 'rho_right', 'xmax', 'ncoord',
             'time_increment', 'max_steps', 'time_interval'},
            set(_panel.check(bad)))
        self.assertEqual({}, _panel.check(SOD))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
