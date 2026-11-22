# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The groups of the reflection control column.

The groups are passive, so these checks hand them a setup or a display and
read the controls back; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.obsrefl import _forms, _setup, _viewmodel
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
        self.numerics = _forms.NumericsGroup()
        self.numerics.show_setup(self.setup)

    def test_the_groups_read_back_the_setup_they_show(self):
        values, errors = self.free.read()
        self.assertEqual({}, errors)
        self.assertEqual(dict(gamma=1.4, density=1.0, pressure=1.0,
                              mach=3.0, angle=10.0), values)
        values, errors = self.numerics.read()
        self.assertEqual({}, errors)
        self.assertEqual(dict(nx=64, ny=16, time_increment=2e-3,
                              cell_type='unstructured'), values)
        self.assertIsInstance(values['nx'], int)
        self.assertEqual(list(_setup.ReflectionSetup.FREE_STREAM),
                         list(self.free.fields))

    def test_a_field_that_holds_no_number_is_named(self):
        _type(self.free, 'mach', "fast")
        values, errors = self.free.read()
        self.assertEqual(['mach'], list(errors))
        self.assertIn("not a number", errors['mach'])
        self.assertNotIn('mach', values)
        _type(self.numerics, 'nx', "inf")
        self.assertEqual(['nx'], list(self.numerics.read()[1]))

    def test_an_edit_reads_dirty_until_shown_again(self):
        self.assertFalse(self.free.is_dirty())
        self.assertFalse(self.numerics.is_dirty())
        _type(self.free, 'angle', "15")
        self.assertTrue(self.free.is_dirty())
        self.numerics._cells.setCurrentText('quad')
        self.assertTrue(self.numerics.is_dirty())
        # Typed back, the edit is gone; text that is no number is an edit.
        _type(self.free, 'angle', "10.0")
        self.assertFalse(self.free.is_dirty())
        _type(self.free, 'angle', "")
        self.assertTrue(self.free.is_dirty())
        applied = self.setup.replace(angle=15.0, cell_type='quad')
        self.free.show_setup(applied)
        self.numerics.show_setup(applied)
        self.assertFalse(self.free.is_dirty())
        self.assertFalse(self.numerics.is_dirty())
        self.assertEqual("15", self.free.fields['angle'].text())

    def test_the_run_settings_apply_at_once(self):
        numerics = self.numerics
        changed = []
        numerics.run_changed = lambda *args: changed.append(args)
        self.assertEqual((2000, 5), numerics.run_settings())
        numerics._cap.setValue(300)
        numerics._frame.setValue(2)
        self.assertEqual([(300, 5), (300, 2)], changed)
        # Nor are they edits that wait for Reset.
        self.assertFalse(numerics.is_dirty())
        numerics.set_run_settings(100, 1)
        self.assertEqual((100, 1), numerics.run_settings())
        self.assertEqual(2, len(changed))


@testing.requires_pilot
class DisplayGroupTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_the_group_reads_back_the_display_it_shows(self):
        group = _forms.DisplayGroup()
        self.assertEqual(_viewmodel.Display(), group.display())
        shown = _viewmodel.Display(field='pressure', bar=False,
                                   mark_cells=True, mark_line=True, cut=0.25)
        group.show_display(shown)
        self.assertEqual(shown, group.display())
        self.assertEqual("pressure", group._field.currentText())

    def test_every_change_reaches_the_owner(self):
        group = _forms.DisplayGroup()
        changed = []
        group.display_changed = changed.append
        group._field.setCurrentIndex(group._field.findData('mach'))
        group._bar.setChecked(False)
        group._cells.setChecked(True)
        group._line.setChecked(True)
        group._cut.setValue(0.75)
        self.assertEqual(5, len(changed))
        self.assertEqual(_viewmodel.Display('mach', False, True, True, 0.75),
                         changed[-1])
        group.show_display(_viewmodel.Display())
        self.assertEqual(5, len(changed))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
