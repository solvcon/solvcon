# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The run strip an app window is headed by.

The strip is passive, so these checks hand it a state and read the controls
back; none needs a window on screen.
"""

import unittest

from solvcon import pilot
from solvcon import testing
from solvcon.pilot import panel


def _state(state, step=0, max_steps=2000, **kw):
    """Return the :class:`StripState` of a run at ``step`` with a time
    increment of 0.002."""
    return panel.StripState(state, step, max_steps, 0.002 * step,
                            0.002 * max_steps, **kw)


def _clock(strip):
    """Return the time and the end time the strip reads."""
    time, t_end = strip._time.text().removeprefix("t = ").split(" / ")
    return float(time), float(t_end)


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
        strip = panel.RunStrip()
        self.addCleanup(testing.dispose_widget, strip)
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
        self.assertEqual(set(panel.RUN_STATES), set(expected))
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
        strip = panel.RunStrip()
        self.addCleanup(testing.dispose_widget, strip)
        strip.show_state(_state('ready', dirty=True))
        reset = strip.buttons['reset']
        self.assertTrue(reset.isEnabled())
        self.assertEqual("true", reset.property("attention"))
        # Edits that cannot build a run keep Reset, and Run again, from
        # trying; the standing run still marches.
        strip.show_state(_state('paused', 5, dirty=True, can_reset=False))
        self.assertEqual((True, True, False), self._enabled(strip))
        self.assertIn("Fix the outlined values", reset.toolTip())
        for state in ('finished', 'failed'):
            with self.subTest(state=state):
                strip.show_state(_state(state, 20, dirty=True,
                                        can_reset=False))
                self.assertEqual((False, False, False), self._enabled(strip))

    def test_the_strip_reads_time_steps_and_progress(self):
        strip = panel.RunStrip()
        self.addCleanup(testing.dispose_widget, strip)
        strip.show_state(_state('running', 500))
        self.assertEqual((1.0, 4.0), _clock(strip))
        self.assertEqual("step 500 / 2000", strip._steps.full_text)
        self.assertEqual(250, strip._progress.value())
        # No cap reads no progress rather than divide by it.
        strip.show_state(_state('ready', 0, 0))
        self.assertEqual(0, strip._progress.value())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
