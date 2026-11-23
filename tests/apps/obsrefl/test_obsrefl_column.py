# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the reflection app.

The widgets are passive, so these checks hand them what to show and read
the controls back; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.obsrefl import _column, _viewmodel
except ImportError:
    pilot = None


def _results(field="density", zones=3):
    """Return the :class:`Results` of a run with ``zones`` zones."""
    cell = _viewmodel.Cell
    rows = tuple(_viewmodel.ResultRow(
        f"zone {it + 1}", cell("1.000"), cell("1.004"),
        cell("+0.42%", 'good', "relative to the exact value"))
        for it in range(zones))
    fits = (_viewmodel.ResultRow("shock angle", cell("29.31"),
                                 cell("29.32"), cell("-0.03%", 'good')),
            _viewmodel.ResultRow("wall x", cell("not formed", 'muted',
                                                "npoint = 1"),
                                 cell("2.851"), cell("", 'muted')))
    return _viewmodel.Results(
        field=field, zones=rows, fits=fits,
        rows=(("mass drift", "+1.2e-9"), ("CFL", "0.41 to 0.52")),
        caption="errors relative to the analytic zone values", note=None)


@testing.requires_pilot
class ResultsCardTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_the_card_fills_every_cell(self):
        card = _column.ResultsCard()
        self.addCleanup(testing.dispose_widget, card)
        card.show_results(_results())
        self.assertEqual(_viewmodel.Cell("+0.42%", 'good',
                                         "relative to the exact value"),
                         card.cell('zones', 2, 2))
        self.assertEqual("1.004", card.cell('zones', 0, 1).text)
        self.assertEqual(_viewmodel.Cell("not formed", 'muted',
                                         "npoint = 1"),
                         card.cell('fits', 1, 0))
        self.assertIsNone(card.cell('zones', 3, 0))
        self.assertEqual("+1.2e-9", card.row_text("mass drift"))
        self.assertEqual("0.41 to 0.52", card.row_text("CFL"))
        self.assertIsNone(card.row_text("energy drift"))
        self.assertEqual("", card.note())

    def test_a_field_without_an_exact_value_says_so(self):
        card = _column.ResultsCard()
        self.addCleanup(testing.dispose_widget, card)
        card.show_results(_results(field="CFL number", zones=0))
        self.assertIsNone(card.cell('zones', 0, 0))
        self.assertIn("CFL number", card._values[('zones', 'empty')].text())
        self.assertEqual("29.31", card.cell('fits', 0, 0).text)
        # Back to a field with zones, the card grows its rows again.
        card.show_results(_results())
        self.assertEqual("1.000", card.cell('zones', 2, 0).text)

    def test_no_run_blanks_the_card(self):
        card = _column.ResultsCard()
        self.addCleanup(testing.dispose_widget, card)
        self.assertEqual("No run", card.note())
        card.show_results(_results())
        card.show_results(None)
        self.assertEqual("No run", card.note())
        self.assertIsNone(card.cell('zones', 0, 0))
        self.assertIsNone(card.row_text("CFL"))


@testing.requires_pilot
class ControlColumnTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_one_group_folds_open_at_a_time(self):
        column = _column.ControlColumn()
        self.addCleanup(testing.dispose_widget, column)
        self.assertEqual([False, False, True],
                         [group.is_open for group in column.groups])
        column.freestream._head.click()
        self.assertEqual([True, False, False],
                         [group.is_open for group in column.groups])
        column.freestream._head.click()
        self.assertEqual([False] * 3,
                         [group.is_open for group in column.groups])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
