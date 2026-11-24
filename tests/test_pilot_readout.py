# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The words the run strip and the results card print: an error reads in
percent with its sign, what is not finite reads "undefined", and the clock
holds its width while the time runs."""

import math
import unittest

from solvcon.pilot import panel


class FormatTC(unittest.TestCase):

    def test_an_error_reads_in_percent_with_its_sign(self):
        self.assertEqual("+1.23%", panel.format_error(0.0123))
        self.assertEqual("-0.50%", panel.format_error(-0.005))

    def test_what_is_not_finite_reads_undefined(self):
        for value in (math.nan, math.inf, -math.inf):
            for format_ in (panel.format_sci, panel.format_number,
                            panel.format_signed,
                            panel.format_error):
                with self.subTest(value=value, format=format_.__name__):
                    self.assertEqual("undefined", format_(value))


class ToneTC(unittest.TestCase):

    def test_an_error_is_toned_by_its_size_either_way(self):
        tone_of = panel.tone_of
        self.assertEqual(['good', 'good', '', 'bad', 'bad', 'muted'],
                         [tone_of(error) for error in (0.0, -0.01, 0.03,
                                                       -0.051, 4.0,
                                                       math.nan)])


class ClockTC(unittest.TestCase):

    def test_the_clock_holds_its_width_while_the_time_runs(self):
        self.assertEqual("t = 0.000 / 4", panel.clock(0.0, 4.0))
        self.assertEqual("t = 1.250 / 4", panel.clock(1.25, 4.0))
        self.assertEqual("t = 12.50 / 25", panel.clock(12.5, 25.0))
        self.assertEqual("t =  0.00 / 25", panel.clock(0.0, 25.0))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
