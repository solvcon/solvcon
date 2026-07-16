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
        # Pin a known variant so the syntax and output colors are
        # deterministic, and restore the shared default afterward.
        self.mgr.set_theme("light")
        self.addCleanup(self.mgr.set_theme, "system")
        # Start each test from a clean primary prompt, abandoning any block a
        # previous test left open in the shared interpreter.
        self.term.resetInput()

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

    def test_incomplete_statement_shows_continuation_prompt(self):
        self._type("for i in range(2):")
        self._key(QtCore.Qt.Key_Return)
        # The current line carries the continuation prompt.
        last_line = self.edit.toPlainText().split("\n")[-1]
        self.assertTrue(last_line.startswith("... "))
        # The next line is auto-indented into the block.
        self.assertEqual(self.term.command, "    ")

    def test_multiline_block_runs_when_closed(self):
        self._type("for i in range(2):")
        self._key(QtCore.Qt.Key_Return)
        # Continue at the auto-indented continuation line.
        self._type("print(i)")
        self._key(QtCore.Qt.Key_Return)
        # A blank line (the pre-filled indent alone) closes and runs it.
        self._key(QtCore.Qt.Key_Return)
        text = self.edit.toPlainText()
        # The body was echoed on a continuation line and the block ran.
        self.assertIn("print(i)", text)
        self.assertRegex(text, r"\.\.\. +print\(i\)")
        self.assertIn("0", text)
        self.assertIn("1", text)
        self.assertTrue(text.endswith(">>> "))

    def test_up_does_not_recall_mid_block(self):
        self.term.command = "recall_me = 1"
        self.term.executeCommand()
        self._type("while False:")
        self._key(QtCore.Qt.Key_Return)
        # Mid-continuation, Up must not overwrite the input with history.
        self._key(QtCore.Qt.Key_Up)
        self.assertEqual(self.term.command, "    ")

    def test_write_to_history_appends_above_the_prompt(self):
        self._type("keep")
        self.term.writeToHistory("external line\n")
        text = self.edit.toPlainText()
        self.assertIn("external line", text)
        # The in-progress input survives the external write.
        self.assertEqual(self.term.command, "keep")

    def test_tab_completes_a_single_match(self):
        # A uniquely named handle leaves exactly one completion, which is
        # inserted directly without a popup.
        self.term.command = "terminal_unique_handle = 1"
        self.term.executeCommand()
        self._type("terminal_unique_han")
        self._key(QtCore.Qt.Key_Tab)
        self.assertEqual(self.term.command, "terminal_unique_handle")

    def test_ctrl_r_recalls_a_matching_command(self):
        self.term.command = "apricot_value = 1"
        self.term.executeCommand()
        self.term.command = "cherry_value = 2"
        self.term.executeCommand()
        self._type("apri")
        _send_key(self.edit, QtCore.Qt.Key_R, "r",
                  QtCore.Qt.ControlModifier)
        self.assertEqual(self.term.command, "apricot_value = 1")

    def test_call_tip_on_open_paren_does_not_disturb_input(self):
        # Typing a call opening keeps the input intact; the tip is advisory.
        self._type("range(")
        self.assertEqual(self.term.command, "range(")

    def test_input_is_highlighted(self):
        self._type("while")
        # The keyword blue defined by the highlighter paints the input block.
        pos = self.edit.toPlainText().rfind("while")
        self.assertIn((0, 0, 180), self._layout_colors(pos))

    def test_committed_transcript_is_not_highlighted(self):
        self.term.command = "pass"
        self.term.executeCommand()
        # The committed keyword carries no highlight after it scrolls up.
        pos = self.edit.toPlainText().find("pass")
        self.assertNotIn((0, 0, 180), self._layout_colors(pos))

    def test_stderr_output_is_red(self):
        self.term.command = "1 / 0"
        self.term.executeCommand()
        text = self.edit.toPlainText()
        color = self._doc_color_at(text.find("ZeroDivisionError"))
        # The light table's error red, red-dominant against the light base.
        self.assertEqual((color.red(), color.green(), color.blue()),
                         (170, 0, 0))

    def test_stdout_output_uses_the_theme_text_color(self):
        self.term.command = "print('marker_out')"
        self.term.executeCommand()
        text = self.edit.toPlainText()
        # rfind lands on the captured output, not the input echo above it.
        color = self._doc_color_at(text.rfind("marker_out"))
        expected = self.edit.palette().color(QtGui.QPalette.Text)
        self.assertEqual((color.red(), color.green(), color.blue()),
                         (expected.red(), expected.green(), expected.blue()))

    def test_output_colors_follow_a_dark_switch(self):
        # Under the dark theme the captured stderr stays red-dominant but
        # takes a distinct, brighter value than the light table's, and stdout
        # takes the dark palette's text color.
        self.mgr.set_theme("light")
        self.term.command = "1 / 0"
        self.term.executeCommand()
        light_err = self._doc_color_at(
            self.edit.toPlainText().find("ZeroDivisionError"))

        self.mgr.set_theme("dark")
        self.term.command = "1 / 0"
        self.term.executeCommand()
        dark_err = self._doc_color_at(
            self.edit.toPlainText().rfind("ZeroDivisionError"))

        for color in (light_err, dark_err):
            self.assertGreater(color.red(), color.green())
            self.assertGreater(color.red(), color.blue())
        self.assertNotEqual(
            (light_err.red(), light_err.green(), light_err.blue()),
            (dark_err.red(), dark_err.green(), dark_err.blue()))

        self.term.command = "print('dark_out')"
        self.term.executeCommand()
        out = self._doc_color_at(self.edit.toPlainText().rfind("dark_out"))
        expected = self.edit.palette().color(QtGui.QPalette.Text)
        self.assertEqual((out.red(), out.green(), out.blue()),
                         (expected.red(), expected.green(), expected.blue()))

    def _doc_color_at(self, index):
        """The stored foreground of the character at ``index``."""
        cursor = QtGui.QTextCursor(self.edit.document())
        cursor.setPosition(index + 1)
        return cursor.charFormat().foreground().color()

    def _layout_colors(self, index):
        """Foreground colors the highlighter laid on the block at ``index``."""
        block = self.edit.document().findBlock(index)
        colors = []
        for rng in block.layout().formats():
            color = rng.format.foreground().color()
            colors.append((color.red(), color.green(), color.blue()))
        return colors

    def term_input_start(self):
        """Position just after the current prompt, read off the document."""
        text = self.edit.toPlainText()
        return text.rfind(">>> ") + len(">>> ")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
