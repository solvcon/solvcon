# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the terminal-style console, RPythonTerminalDockWidget.

The terminal presents the in-process interpreter on a single surface: one
document where the prompt, the typed command, and the captured output
interleave, with the committed transcript read-only and the current input
editable after the prompt. These drive the widget through synthesized key
events, so they need the Qt pilot but no on-screen rendering.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None


def _send_key(widget, key, text="", mod=None):
    """Post a synthetic key press and release to ``widget``.

    Built by hand rather than through ``QtTest`` so the tests need only the
    always-present PySide6 widget modules, matching the other pilot tests.
    """
    if mod is None:
        mod = QtCore.Qt.NoModifier
    for etype in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
        event = QtGui.QKeyEvent(etype, key, mod, text)
        QtWidgets.QApplication.sendEvent(widget, event)


@unittest.skipUnless(solvcon.HAS_PILOT and pilot is not None,
                     "Qt pilot is not built")
class TerminalWidgetTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        self.term = self.mgr.pyterm
        self.edit = self.term.textEdit
        # Start each test from an empty input region after the prompt.
        self.term.command = ""

    def _type(self, text):
        for ch in text:
            _send_key(self.edit, ord(ch.upper()), ch)

    def _key(self, key, mod=None):
        _send_key(self.edit, key, "", mod)

    def test_terminal_exists_and_shows_prompt(self):
        self.assertIsNotNone(self.term)
        self.assertIsNotNone(self.edit)
        self.assertTrue(self.edit.toPlainText().endswith(">>> "))

    def test_terminal_greets_with_the_console_banner(self):
        # The terminal opens with the same handles-in-scope banner the
        # two-pane console prints, above its first prompt.
        self.assertIn("solvcon pilot console. Handles in scope:",
                      self.edit.toPlainText())

    def test_typed_text_becomes_the_input(self):
        self._type("1 + 1")
        self.assertEqual(self.term.command, "1 + 1")

    def test_execute_bare_expression_shows_result(self):
        self.term.command = "6 * 7"
        self.term.executeCommand()
        text = self.edit.toPlainText()
        self.assertIn(">>> 6 * 7", text)
        self.assertIn("42", text)
        # A fresh prompt awaits the next command.
        self.assertTrue(text.endswith(">>> "))
        self.assertEqual(self.term.command, "")

    def test_state_persists_across_commands(self):
        self.term.command = "term_x = 5"
        self.term.executeCommand()
        self.term.command = "term_x + 100"
        self.term.executeCommand()
        self.assertIn("105", self.edit.toPlainText())

    def test_home_lands_after_the_prompt(self):
        self._type("abc")
        self._key(QtCore.Qt.Key_Home)
        cursor = self.edit.textCursor()
        self.assertEqual(cursor.position(), self.term_input_start())
        # Typing at Home still edits the input, not the prompt.
        self._type("Z")
        self.assertEqual(self.term.command, "Zabc")

    def test_backspace_stops_at_the_prompt(self):
        self._type("q")
        self._key(QtCore.Qt.Key_Backspace)
        self.assertEqual(self.term.command, "")
        # One more backspace must not eat into the prompt.
        self._key(QtCore.Qt.Key_Backspace)
        self.assertTrue(self.edit.toPlainText().endswith(">>> "))

    def test_typing_in_the_readonly_head_relocates_to_input(self):
        self._type("tail")
        # Move the caret into the committed head.
        cursor = self.edit.textCursor()
        cursor.setPosition(0)
        self.edit.setTextCursor(cursor)
        self._type("X")
        # The character lands in the input region, not the frozen head.
        self.assertEqual(self.term.command, "tailX")

    def test_history_recall_with_up_and_down(self):
        self.term.command = "history_one = 1"
        self.term.executeCommand()
        self.term.command = "history_two = 2"
        self.term.executeCommand()
        self._key(QtCore.Qt.Key_Up)
        self.assertEqual(self.term.command, "history_two = 2")
        self._key(QtCore.Qt.Key_Up)
        self.assertEqual(self.term.command, "history_one = 1")
        self._key(QtCore.Qt.Key_Down)
        self.assertEqual(self.term.command, "history_two = 2")

    def test_write_to_history_appends_above_the_prompt(self):
        self._type("keep")
        self.term.writeToHistory("external line\n")
        text = self.edit.toPlainText()
        self.assertIn("external line", text)
        # The in-progress input survives the external write.
        self.assertEqual(self.term.command, "keep")

    def term_input_start(self):
        """Position just after the current prompt, read off the document."""
        text = self.edit.toPlainText()
        return text.rfind(">>> ") + len(">>> ")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
