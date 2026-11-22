# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The view records of the reflection app and the words they print: an
error reads in percent with its sign, and what is not finite reads
"undefined"."""

import math
import unittest

from solvcon.apps.obsrefl import _viewmodel
from solvcon.multidim.euler import EulerField


class FormatTC(unittest.TestCase):

    def test_an_error_reads_in_percent_with_its_sign(self):
        self.assertEqual("+1.23%", _viewmodel.format_error(0.0123))
        self.assertEqual("-0.50%", _viewmodel.format_error(-0.005))

    def test_what_is_not_finite_reads_undefined(self):
        for value in (math.nan, math.inf, -math.inf):
            for format_ in (_viewmodel.format_sci, _viewmodel.format_number,
                            _viewmodel.format_signed,
                            _viewmodel.format_error):
                with self.subTest(value=value, format=format_.__name__):
                    self.assertEqual("undefined", format_(value))


class ToneTC(unittest.TestCase):

    def test_an_error_is_toned_by_its_size_either_way(self):
        tone_of = _viewmodel.tone_of
        self.assertEqual(['good', 'good', '', 'bad', 'bad', 'muted'],
                         [tone_of(error) for error in (0.0, -0.01, 0.03,
                                                       -0.051, 4.0,
                                                       math.nan)])


class RecordTC(unittest.TestCase):

    def test_every_field_of_the_solver_has_a_label(self):
        self.assertEqual(EulerField.FIELDS,
                         tuple(_viewmodel.FIELD_LABELS))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
