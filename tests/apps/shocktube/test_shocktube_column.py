# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the shock-tube app.

The widgets are passive, so these checks hand them what to show and read
the controls back; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.shocktube import _column, _setup, _viewmodel
except ImportError:
    pilot = None


@testing.requires_pilot
class GroupsTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.column = _column.ControlColumn()
        self.addCleanup(testing.dispose_widget, self.column)

    def test_the_groups_read_what_the_setup_holds(self):
        setup = _setup.ShockTubeSetup()
        problem, numerics = self.column.problem, self.column.numerics
        problem.show_setup(setup)
        numerics.show_setup(setup)
        values, errors = problem.read()
        self.assertEqual({}, errors)
        self.assertEqual(setup.p_right, values['p_right'])
        self.assertEqual(201, numerics.read()[0]['ncoord'])
        self.assertIsInstance(numerics.read()[0]['ncoord'], int)
        self.assertFalse(problem.is_dirty())
        problem.fields['gamma'].setText("abc")
        self.assertEqual({'gamma': "Gamma is not a number."},
                         problem.read()[1])
        self.assertTrue(problem.is_dirty())
        self.assertEqual((50, 1), numerics.run_settings())

    def test_the_display_group_holds_a_display(self):
        display = self.column.display
        self.assertEqual(_viewmodel.Display(), display.display())
        shown = []
        display.display_changed = shown.append
        display.option_boxes['log'].setChecked(True)
        display.quantity_boxes['entropy'].setChecked(True)
        self.assertEqual([True, True], [item.log for item in shown])
        self.assertIn('entropy', shown[-1].quantities)
        # Set from the owner, the group calls nothing back.
        display.show_display(_viewmodel.Display(exact='cells'))
        self.assertEqual('cells', display.display().exact)
        self.assertEqual(2, len(shown))

    def test_one_group_folds_open_at_a_time(self):
        column = self.column
        self.assertEqual([False, False, True],
                         [group.is_open for group in column.groups])
        column.problem._head.click()
        self.assertEqual([True, False, False],
                         [group.is_open for group in column.groups])

    def test_the_card_says_what_it_cannot_judge(self):
        card = self.column.card
        card.show_results(_viewmodel.Results(
            states=(), fronts=(), rows=(("mass drift", "0"),),
            caption="", note=("vacuum", 'muted')))
        self.assertIn("No star state",
                      card._values[('states', 'empty')].text())
        self.assertEqual("vacuum", card.note())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
