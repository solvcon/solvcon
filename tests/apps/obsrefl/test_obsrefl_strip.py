# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The run keys of the reflection window and of its control column.

Keys reach the widgets as posted events; none needs a window on screen.
"""

import unittest

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QApplication, QWidget
        from solvcon.apps.obsrefl import _column, _window
except ImportError:
    pilot = None


def _state(state, step=0, max_steps=2000, **kw):
    """Return the :class:`StripState` of a run at ``step`` with a time
    increment of 0.002."""
    return pilot.panel.StripState(state, step, max_steps, 0.002 * step,
                                  0.002 * max_steps, **kw)


def _key(widget, key, text="", modifiers=None, repeat=False):
    """Press and release ``key`` on ``widget``."""
    modifiers = Qt.NoModifier if modifiers is None else modifiers
    # Posted rather than sent: a key sent into the pilot binary from a
    # test can crash it.
    for kind in (QEvent.KeyPress, QEvent.KeyRelease):
        QApplication.postEvent(widget, QKeyEvent(kind, key, modifiers, text,
                                                 repeat))
    QApplication.processEvents()


@testing.requires_pilot
class RunKeysTC(unittest.TestCase):
    """The run keys press the enabled strip buttons from the window and
    from the docked column, and leave a key a focused control takes."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.window = _window.ReflectionWindow(QWidget())
        self.column = _column.ControlColumn()
        self.column.key_pressed = self.window.press_key
        self.pressed = []
        strip = self.window.strip
        for name in ('run', 'step', 'reset'):
            setattr(strip, f"{name}_clicked",
                    lambda name=name: self.pressed.append(name))
        strip.show_state(_state('paused', 10))

    def tearDown(self):
        testing.dispose_widget(self.window)
        testing.dispose_widget(self.column)

    def test_the_run_keys_press_the_strip_buttons(self):
        window = self.window
        _key(window, Qt.Key_Space, " ")
        _key(window, Qt.Key_Right)
        _key(window, Qt.Key_R, "r")
        self.assertEqual(['run', 'step', 'reset'], self.pressed)

    def test_a_disabled_button_takes_no_key(self):
        self.window.strip.show_state(_state('running', 10))
        _key(self.window, Qt.Key_Right)
        self.assertEqual([], self.pressed)
        self.window.strip.show_state(_state('ready'))
        _key(self.window, Qt.Key_R, "r")
        self.assertEqual([], self.pressed)

    def test_a_held_key_acts_once_but_step_repeats(self):
        window = self.window
        _key(window, Qt.Key_Space, " ", repeat=True)
        _key(window, Qt.Key_R, "r", repeat=True)
        self.assertEqual([], self.pressed)
        _key(window, Qt.Key_Right, repeat=True)
        _key(window, Qt.Key_Right, repeat=True)
        self.assertEqual(['step', 'step'], self.pressed)

    def test_a_modified_key_is_the_pilots(self):
        _key(self.window, Qt.Key_Right, modifiers=Qt.ControlModifier)
        _key(self.window, Qt.Key_R, "r", modifiers=Qt.ShiftModifier)
        self.assertEqual([], self.pressed)
        # The keypad sends its keys with a modifier of its own.
        _key(self.window, Qt.Key_Right, modifiers=Qt.KeypadModifier)
        self.assertEqual(['step'], self.pressed)

    def test_the_keys_reach_the_strip_from_the_column(self):
        _key(self.column.card, Qt.Key_Space, " ")
        _key(self.column, Qt.Key_Right)
        self.assertEqual(['run', 'step'], self.pressed)

    def test_a_focused_control_keeps_its_keys(self):
        box = self.column.display._bar
        _key(box, Qt.Key_Space, " ")
        self.assertFalse(box.isChecked())
        _key(self.column.numerics._cap, Qt.Key_Right)
        _key(self.column.display._cut, Qt.Key_Right)
        _key(self.column.freestream.fields['mach'], Qt.Key_Right)
        self.assertEqual([], self.pressed)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
