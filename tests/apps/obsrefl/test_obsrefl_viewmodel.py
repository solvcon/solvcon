# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The view records of the reflection app."""

import unittest

from solvcon.apps.obsrefl import _viewmodel
from solvcon.multidim.euler import EulerField


class RecordTC(unittest.TestCase):

    def test_every_field_of_the_solver_has_a_label(self):
        self.assertEqual(EulerField.FIELDS,
                         tuple(_viewmodel.FIELD_LABELS))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
