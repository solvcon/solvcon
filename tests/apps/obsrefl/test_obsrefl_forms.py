# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the reflection control column.

The groups are passive, so these checks hand them a setup and read the
controls back; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.obsrefl import _forms, _setup
except ImportError:
    pilot = None


def _type(group, key, text):
    """Type ``text`` into the field ``key`` of ``group``."""
    field = group.fields[key]
    field.setText(text)
    field.textEdited.emit(text)


@testing.requires_pilot
class SetupGroupTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.setup = _setup.ReflectionSetup()
        self.free = _forms.FreeStreamGroup()
        self.free.show_setup(self.setup)

    def test_the_groups_read_back_the_setup_they_show(self):
        values, errors = self.free.read()
        self.assertEqual({}, errors)
        self.assertEqual(dict(gamma=1.4, density=1.0, pressure=1.0,
                              mach=3.0, angle=10.0), values)
        self.assertEqual(list(_setup.ReflectionSetup.FREE_STREAM),
                         list(self.free.fields))

    def test_a_field_that_holds_no_number_is_named(self):
        _type(self.free, 'mach', "fast")
        values, errors = self.free.read()
        self.assertEqual(['mach'], list(errors))
        self.assertIn("not a number", errors['mach'])
        self.assertNotIn('mach', values)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
