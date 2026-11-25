# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The controller of the shock-tube app over the real window and column.

The widgets are built but not shown, and the frame timer is driven by
direct ticks, so these checks march a run as fast as it goes and read the
widgets back.
"""

import unittest
from unittest import mock

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.shocktube import (_column, _control, _figure,
                                            _setup, _window)
except ImportError:
    pilot = None


@testing.requires_pilot
class ShockTubeControllerTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.window = _window.ShockTubeWindow(_figure.ShockTubeFigure())
        self.column = _column.ControlColumn()
        self.addCleanup(testing.dispose_widget, self.window)
        self.addCleanup(testing.dispose_widget, self.column)
        self.ctl = _control.ShockTubeController(
            self.window, self.column, _setup.ShockTubeSetup(ncoord=41))
        self.strip = self.window.strip

    def test_a_new_cap_rescales_the_xt_diagram_at_once(self):
        self.ctl.step()
        xt = self.window.figure.xt
        self.assertAlmostEqual(2.5, xt.t_end)
        self.column.numerics.set_run_settings(500, 1)
        self.column.numerics._on_run(None)
        # The diagram reaches up to the time of the new cap without waiting
        # for another frame.
        self.assertAlmostEqual(25.0, xt.t_end)
        self.assertEqual('paused', self.ctl.state)
        # A cap under the step the run stands at ends it there.
        self.column.numerics.set_run_settings(1, 1)
        self.column.numerics._on_run(None)
        self.assertEqual('finished', self.ctl.state)
        self.assertAlmostEqual(0.05, xt.t_end)

    def test_a_march_gone_wrong_fails_the_run(self):
        self.ctl.step()
        values = self.ctl.session.values()
        values['density'][3] = float('nan')
        with mock.patch.object(type(self.ctl.session), 'values',
                               return_value=values), \
                mock.patch('sys.excepthook'):
            self.ctl.step()
        self.assertEqual('failed', self.ctl.state)
        self.assertEqual(("Failed", 'bad'), self.strip.pill())
        self.assertIn("no longer finite", self.strip._failure.full_text)
        # Run again builds the run anew and marches it.
        self.strip.buttons['run'].click()
        self.assertEqual(('running', 0), (self.ctl.state,
                                          self.ctl.session.step))
        self.ctl.pause()

    def test_a_setup_the_solver_cannot_build_keeps_the_run(self):
        self.ctl.step()
        standing = self.ctl.session
        with mock.patch.object(_setup.ShockTubeSetup, 'session',
                               side_effect=RuntimeError("boom")):
            self.assertFalse(self.ctl.reset())
        self.assertIs(standing, self.ctl.session)
        self.assertIn("No run can be built of Sod, 41 points: boom",
                      self.strip._failure.full_text)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
