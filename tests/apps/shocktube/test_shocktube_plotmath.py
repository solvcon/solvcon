# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The arithmetic of the shock-tube plots, which needs no Qt."""

import unittest

from solvcon.apps import shocktube
from solvcon.apps.shocktube import _plotmath


class PlotMathTC(unittest.TestCase):

    def test_a_log_axis_spaces_the_decades_evenly(self):
        axis = _plotmath.Axis(-2.0, 2.0, 100.0, 0.0, log=True)
        self.assertAlmostEqual(25.0, axis(10.0))

    def test_a_label_gives_way_to_one_placed_before_it(self):
        labels = [(50, 20, (0,)), (55, 20, (0,))]
        self.assertEqual([(0, 0, 40.0)], _plotmath.place(labels, 0, 200))

    def test_sod_reads_left_fan_star_right(self):
        exact = shocktube.ExactRiemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                       gamma=1.4)
        spans = _plotmath.region_spans(exact.positions(0.2), -1.0, 1.0)
        self.assertEqual(['L', 'fan', 'L*', 'R*', 'R'],
                         [name for name, _, _ in spans])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
