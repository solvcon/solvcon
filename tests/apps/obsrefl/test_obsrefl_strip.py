# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The run strip of the reflection app.

The strip is passive, so these checks hand it a state and read the controls
back; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps.obsrefl import _viewmodel, _window
except ImportError:
    pilot = None


def _state(state, step=0, max_steps=2000, **kw):
    """Return the :class:`StripState` of a run at ``step`` with a time
    increment of 0.002."""
    return _viewmodel.StripState(state, step, max_steps, 0.002 * step,
                                 0.002 * max_steps, **kw)


@testing.requires_pilot
class RunStripTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    @staticmethod
    def _enabled(strip):
        return tuple(strip.buttons[name].isEnabled()
                     for name in ('run', 'step', 'reset'))

    def test_the_buttons_follow_what_the_run_can_still_do(self):
        strip = _window.RunStrip()
        expected = {
            # A fresh run has nothing to reset yet.
            'ready': ("Run", (True, True, False), ("Ready", 'muted')),
            'running': ("Pause", (True, False, True),
                        ("Running", 'accent')),
            'paused': ("Resume", (True, True, True), ("Paused", 'muted')),
            # An ended run steps no further; the accent runs it again.
            'finished': ("Run again", (True, False, True),
                         ("Finished", 'good')),
            'failed': ("Run again", (True, False, True), ("Failed", 'bad')),
        }
        self.assertEqual(set(_viewmodel.STATES), set(expected))
        for state, (text, enabled, pill) in expected.items():
            with self.subTest(state=state):
                step = 0 if 'ready' == state else 10
                strip.show_state(_state(state, step))
                self.assertEqual(state, strip.state)
                self.assertEqual(text, strip.buttons['run'].text())
                self.assertEqual(enabled, self._enabled(strip))
                self.assertEqual(pill, strip.pill())
                self.assertEqual("false",
                                 strip.buttons['reset'].property("attention"))

    def test_edits_waiting_for_reset_draw_the_eye(self):
        strip = _window.RunStrip()
        strip.show_state(_state('ready', dirty=True))
        reset = strip.buttons['reset']
        self.assertTrue(reset.isEnabled())
        self.assertEqual("true", reset.property("attention"))
        # Edits that cannot build a run keep Reset, and Run again, from
        # trying; the standing run still marches.
        strip.show_state(_state('paused', 5, dirty=True, can_reset=False))
        self.assertEqual((True, True, False), self._enabled(strip))
        for state in ('finished', 'failed'):
            with self.subTest(state=state):
                strip.show_state(_state(state, 20, dirty=True,
                                        can_reset=False))
                self.assertEqual((False, False, False), self._enabled(strip))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
