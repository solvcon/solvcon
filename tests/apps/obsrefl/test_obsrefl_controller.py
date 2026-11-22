# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The controller of the reflection app over the real window and column.

The widgets are built but not shown, and the frame timer is driven by
direct ticks, so these checks march a coarse run as fast as it goes and
read the widgets back.  The domain view draws only on a live surface;
nothing here reads its pixels.
"""

import unittest
from unittest import mock

import numpy as np

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QApplication
        from solvcon.apps.obsrefl import (_column, _control, _figure,
                                          _presenter, _setup, _window)
except ImportError:
    pilot = None


def _type(field, text):
    """Type ``text`` into the number ``field`` as the user would."""
    field.setText(text)
    field.textEdited.emit(text)


def _key(widget, key, text=""):
    """Press and release ``key`` on ``widget``.  QtTest is not in every
    PySide6 build, so the events are sent as it would send them."""
    for kind in (QEvent.KeyPress, QEvent.KeyRelease):
        QApplication.sendEvent(widget, QKeyEvent(kind, key, Qt.NoModifier,
                                                 text))


@testing.requires_pilot
class ReflectionControllerTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        self.figure = _figure.ReflectionFigure(mgr=self.mgr)
        self.window = _window.ReflectionWindow(self.figure)
        self.column = _column.ControlColumn()
        self.addCleanup(testing.dispose_widget, self.window)
        self.addCleanup(testing.dispose_widget, self.column)
        self.ctl = _control.ReflectionController(
            self.window, self.column, _setup.ReflectionSetup(nx=10, ny=4))
        self.addCleanup(self.ctl.shutdown)
        self.strip = self.window.strip

    def _enabled(self):
        return {name: button.isEnabled()
                for name, button in self.strip.buttons.items()}

    def test_opening_previews_the_run_at_step_0(self):
        ctl = self.ctl
        self.assertEqual('ready', ctl.state)
        self.assertEqual(0, ctl.session.step)
        self.assertFalse(ctl._timer.isActive())
        self.assertIn("10 x 4", self.strip.problem())
        self.assertTrue(self.figure.caption.text().startswith(
            "density, step 0:"))
        self.assertTrue(self.column.card.row_text('CFL'))

    def test_run_marches_a_frame_a_tick_and_pauses(self):
        ctl = self.ctl
        self.column.numerics._frame.setValue(3)
        self.strip.buttons['run'].click()
        self.assertEqual('running', ctl.state)
        self.assertTrue(ctl._timer.isActive())
        self.assertFalse(self._enabled()['step'])
        ctl.step()
        self.assertEqual(0, ctl.session.step)
        ctl._tick()
        ctl._tick()
        self.assertEqual(6, ctl.session.step)
        self.assertTrue(self.figure.caption.text().startswith(
            "density, step 6:"))
        ctl.run_or_pause()
        self.assertEqual('paused', ctl.state)
        self.assertFalse(ctl._timer.isActive())
        ctl.step()
        self.assertEqual(9, ctl.session.step)
        self.assertEqual('paused', ctl.state)

    def test_the_cap_finishes_the_run_and_run_again_rebuilds(self):
        ctl = self.ctl
        self.column.numerics._cap.setValue(4)
        self.column.numerics._frame.setValue(3)
        ctl.run()
        ctl._tick()
        ctl._tick()
        self.assertEqual(4, ctl.session.step)
        self.assertEqual('finished', ctl.state)
        self.assertFalse(ctl._timer.isActive())
        first = ctl.session
        ctl.run_or_pause()
        self.assertIsNot(first, ctl.session)
        self.assertEqual(0, ctl.session.step)
        self.assertEqual('running', ctl.state)

    def test_the_cap_applies_to_the_standing_run_at_once(self):
        ctl = self.ctl
        ctl.step()
        ctl.step()
        self.assertEqual(10, ctl.session.step)
        self.column.numerics._cap.setValue(12)
        self.assertEqual(12, ctl.session.max_steps)
        ctl.step()
        self.assertEqual(12, ctl.session.step)
        self.assertEqual('finished', ctl.state)
        # A cap under the step the run stands at ends the run there.
        ctl.reset()
        ctl.step()
        self.column.numerics._cap.setValue(3)
        self.assertEqual('finished', ctl.state)
        self.assertEqual(5, ctl.session.step)

    def test_a_higher_cap_lets_a_finished_run_go_on(self):
        ctl = self.ctl
        numerics = self.column.numerics
        numerics._cap.setValue(5)
        ctl.step()
        self.assertEqual('finished', ctl.state)
        session = ctl.session
        numerics._cap.setValue(20)
        self.assertEqual('paused', ctl.state)
        self.assertTrue(self._enabled()['step'])
        ctl.step()
        self.assertIs(session, ctl.session)
        self.assertEqual(10, session.step)

    def test_a_cap_change_keeps_a_failed_run_failed(self):
        ctl = self.ctl
        ctl.run()
        with mock.patch.object(ctl.session, 'advance',
                               side_effect=RuntimeError("blew up")), \
                mock.patch('sys.excepthook'):
            ctl._tick()
        self.column.numerics._cap.setValue(1)
        self.assertEqual('failed', ctl.state)
        self.assertEqual(('Failed', 'bad'), self.strip.pill())

    def test_a_cap_applies_once_typed_whole(self):
        ctl = self.ctl
        ctl.step()
        ctl.step()
        cap = self.column.numerics._cap
        cap.lineEdit().selectAll()
        for digit in "5000":
            # The code of a digit key is the digit's character code.
            _key(cap, ord(digit), digit)
        self.assertEqual('paused', ctl.state)
        self.assertEqual(2000, ctl.session.max_steps)
        _key(cap, Qt.Key_Return)
        self.assertEqual(5000, ctl.session.max_steps)
        self.assertEqual('paused', ctl.state)

    def test_an_invalid_mach_keeps_the_standing_run(self):
        ctl = self.ctl
        ctl.step()
        session = ctl.session
        freestream, numerics = self.column.freestream, self.column.numerics
        _type(freestream.fields['mach'], "0.9")
        _type(numerics.fields['nx'], "12.5")
        self.assertEqual(['mach'], freestream.invalid_fields())
        self.assertEqual(['nx'], numerics.invalid_fields())
        self.assertIn("1.1 to 20", freestream.error_text())
        self.assertNotIn("nx", freestream.error_text())
        self.assertFalse(self._enabled()['reset'])
        self.assertFalse(ctl.reset())
        self.assertIs(session, ctl.session)
        self.assertEqual('paused', ctl.state)
        _type(numerics.fields['nx'], "10")
        _type(freestream.fields['mach'], "3")
        _type(freestream.fields['angle'], "40")
        self.assertEqual(['angle'], freestream.invalid_fields())
        self.assertFalse(self._enabled()['reset'])

    def test_a_valid_edit_waits_for_reset(self):
        ctl = self.ctl
        titles = []
        ctl.title_changed = titles.append
        freestream = self.column.freestream
        _type(freestream.fields['angle'], "15")
        self.assertTrue(freestream.summary().endswith("(pending)"))
        self.assertEqual("true",
                         self.strip.buttons['reset'].property("attention"))
        self.assertEqual(10, ctl.setup.angle)
        _type(self.column.numerics.fields['nx'], "12")
        self.assertTrue(self.column.numerics.summary().endswith("(pending)"))
        self.strip.buttons['reset'].click()
        self.assertEqual(15, ctl.setup.angle)
        self.assertEqual(12, ctl.setup.nx)
        self.assertEqual('ready', ctl.state)
        self.assertFalse(freestream.summary().endswith("(pending)"))
        self.assertEqual("false",
                         self.strip.buttons['reset'].property("attention"))
        self.assertEqual(1, len(titles))
        self.assertIn("15 deg, 12 x 4", titles[0])

    def test_the_display_applies_without_marching(self):
        ctl = self.ctl
        ctl.step()
        display = self.column.display
        display._field.setCurrentIndex(display._field.findData('pressure'))
        self.assertEqual(5, ctl.session.step)
        self.assertTrue(self.figure.caption.text().startswith("pressure"))
        self.assertTrue(display.summary().startswith("pressure"))
        display._bar.setChecked(False)
        self.assertTrue(self.figure.bar.isHidden())
        display._cut.setValue(0.25)
        np.testing.assert_array_equal(
            _presenter.profile_frame(ctl.session, ctl.display).computed,
            self.figure.profile.frame.computed)
        self.assertEqual(5, ctl.session.step)

    def test_a_raising_march_fails_the_run(self):
        ctl = self.ctl
        ctl.run()
        with mock.patch.object(ctl.session, 'advance',
                               side_effect=RuntimeError("blew up")), \
                mock.patch('sys.excepthook'):
            ctl._tick()
        self.assertEqual('failed', ctl.state)
        self.assertFalse(ctl._timer.isActive())
        self.assertIn("blew up", self.strip._failure.full_text)
        ctl.run_or_pause()
        self.assertEqual('running', ctl.state)
        self.assertEqual('', ctl.error)

    def test_a_build_the_solver_rejects_keeps_the_standing_run(self):
        ctl = self.ctl
        ctl.step()
        session = ctl.session
        _type(self.column.freestream.fields['angle'], "15")
        with mock.patch.object(_setup.ReflectionSetup, 'session',
                               side_effect=RuntimeError("no mesh")):
            self.assertFalse(ctl.reset())
        self.assertIs(session, ctl.session)
        self.assertEqual('paused', ctl.state)
        self.assertEqual(10, ctl.setup.angle)
        self.assertIn("no mesh", self.strip._failure.full_text)

    def test_a_density_no_longer_finite_fails_the_run(self):
        ctl = self.ctl
        session = ctl.session
        march = session.advance

        def advance():
            march()
            session.field.conserved[0, 0] = np.nan

        ctl.run()
        with mock.patch.object(session, 'advance', side_effect=advance), \
                mock.patch('sys.excepthook'):
            ctl._tick()
        self.assertEqual('failed', ctl.state)
        self.assertFalse(ctl._timer.isActive())
        self.assertIn("finite", self.strip._failure.full_text)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
