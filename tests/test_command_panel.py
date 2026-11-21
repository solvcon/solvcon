# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the command panel's pieces that need no live window: the
Enter rule, the cell runner, and the composer's keys.
"""

import gc
import unittest
from unittest import mock

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot.panel import _command
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None


@testing.requires_pilot
class EnterRuleTC(unittest.TestCase):

    def test_open_block_asks_for_more(self):
        self.assertFalse(_command.python_is_complete(
            "for i in range(3):"))
        self.assertFalse(_command.python_is_complete(
            "for i in range(3):\n    print(i)"))

    def test_blank_last_line_closes_the_block(self):
        self.assertTrue(_command.python_is_complete(
            "for i in range(3):\n    print(i)\n"))

    def test_a_blank_line_does_not_close_a_string_or_a_bracket(self):
        for source in ('s = """\n', 'f(\n', 'f(1,\n    ', 's = """\n    '):
            with self.subTest(source=source):
                self.assertFalse(_command.python_is_complete(source))

    def test_broken_source_runs_to_show_the_error(self):
        self.assertTrue(_command.python_is_complete("x = )"))


@testing.requires_pilot
class RunCellTC(unittest.TestCase):

    def _text(self, runs, is_error=None):
        return "".join(t for e, t in runs if is_error is None or e == is_error)

    def test_dedented_statement_after_a_block_runs(self):
        # Pushed line by line into the interactive console this is a
        # SyntaxError; as a cell it runs like a file.
        runs = _command.run_cell(
            "for i in range(2):\n    print(i)\nprint('done')")
        self.assertEqual(self._text(runs), "0\n1\ndone\n")

    def test_stdout_and_stderr_keep_their_order(self):
        runs = _command.run_cell(
            "import sys\nprint('a')\nprint('b', file=sys.stderr)\n"
            "print('c')")
        self.assertEqual(runs, [(False, "a\n"), (True, "b\n"),
                                (False, "c\n")])

    def test_stderr_output_alone_is_not_an_exception(self):
        runs = _command.run_cell(
            "import sys\nprint('careful', file=sys.stderr)")
        self.assertIn("careful", self._text(runs, True))
        self.assertFalse(runs.raised)
        self.assertTrue(_command.run_cell("1 / 0").raised)
        self.assertTrue(_command.run_cell("1 +").raised)

    def test_state_persists_in_the_console_namespace(self):
        _command.run_cell("_cp_value = 5")
        self.assertEqual(
            self._text(_command.run_cell("_cp_value + 1")), "6\n")

    def test_input_reads_eof(self):
        err = self._text(_command.run_cell("input()"), True)
        self.assertIn("EOFError", err)

    def test_exit_is_reported_and_the_pilot_lives_on(self):
        runs = _command.run_cell("raise SystemExit(2)")
        self.assertTrue(runs.raised)
        self.assertIn("use the window controls", self._text(runs, True))

    def test_a_failing_cell_gives_the_streams_back(self):
        import sys
        streams = (sys.stdin, sys.stdout, sys.stderr)
        for source in ("raise SystemExit(2)", "x = )", "1 / 0"):
            with self.subTest(source=source):
                _command.run_cell(source)
                self.assertEqual((sys.stdin, sys.stdout, sys.stderr),
                                 streams)


@testing.requires_pilot
class MessageSinkTC(unittest.TestCase):

    def test_messages_wait_for_the_sink(self):
        from solvcon.pilot.base import _gui_common
        with mock.patch.object(_gui_common, "_message_sink", None), \
                mock.patch.object(_gui_common, "_pending_messages", []):
            _gui_common.write_message("early\n")
            shown = []
            _gui_common.set_message_sink(shown.append)
            _gui_common.write_message("late\n")
            self.assertEqual(shown, ["early\n", "late\n"])


@testing.requires_pilot
class PanelStyleTC(unittest.TestCase):

    def setUp(self):
        pilot.RManager.instance.setUp()

    def test_listing_blocks_leaves_no_wrapper_of_a_layout_item(self):
        # Such a wrapper outlives its item, and a widget allocated at the same
        # address would then read as a QWidgetItem.
        def wrappers():
            gc.collect()
            return sum(isinstance(o, QtWidgets.QWidgetItem)
                       for o in gc.get_objects())

        panel = _command.CommandPanelWidget()
        kept = panel.add_block(_command.PYTHON, "x = 1")
        gone = panel.add_block(_command.PYTHON, "y = 2")
        before = wrappers()
        self.assertEqual(panel.blocks(), [kept, gone])
        self.assertEqual(wrappers(), before)
        panel.clear(keep=kept)
        self.assertEqual(panel.blocks(), [kept])


@testing.requires_pilot
class ComposerTC(unittest.TestCase):

    def setUp(self):
        pilot.RManager.instance.setUp()
        self.composer = _command.CommandComposer()
        self.submitted = []
        self.composer.submitted.connect(self.submitted.append)

    def _key(self, key, text="", mod=None, repeat=False):
        if mod is None:
            mod = QtCore.Qt.NoModifier
        event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, mod, text,
                                repeat)
        QtWidgets.QApplication.sendEvent(self.composer, event)

    def test_ctrl_i_toggles_the_mode(self):
        self._key(QtCore.Qt.Key_I, "", QtCore.Qt.ControlModifier)
        self.assertEqual(self.composer.mode, _command.AGENT)
        self._key(QtCore.Qt.Key_I, "", QtCore.Qt.ControlModifier)
        self.assertEqual(self.composer.mode, _command.PYTHON)

    def test_backspace_on_empty_input_switches_back(self):
        self.composer.set_mode(_command.AGENT)
        self._key(QtCore.Qt.Key_Backspace)
        self.assertEqual(self.composer.mode, _command.PYTHON)

    def test_enter_on_an_open_block_indents(self):
        self.composer.set_text("for i in range(3):")
        self._key(QtCore.Qt.Key_Return, "\r")
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.composer.toPlainText(),
                         "for i in range(3):\n    ")

    def test_enter_on_a_complete_statement_submits(self):
        self.composer.set_text("x = 1")
        self._key(QtCore.Qt.Key_Return, "\r")
        self.assertEqual(self.submitted, ["x = 1"])

    def test_shift_enter_always_opens_a_line(self):
        self.composer.set_text("x = 1")
        self._key(QtCore.Qt.Key_Return, "\r", QtCore.Qt.ShiftModifier)
        self.assertEqual(self.submitted, [])


@testing.requires_pilot
class BlockWidthTC(unittest.TestCase):
    """A run with no space to wrap at must not widen the transcript, or
    every block's status slides out of the dock."""

    def setUp(self):
        pilot.RManager.instance.setUp()
        self.panel = _command.CommandPanelWidget()
        self.addCleanup(self.panel.deleteLater)
        self.narrow = self.panel.add_block(
            _command.PYTHON, "x = 1").minimumSizeHint().width()

    def test_long_output_keeps_the_block_narrow(self):
        block = self.panel.add_block(_command.PYTHON, "x = 1")
        block.add_output("x" * 1000)
        self.assertEqual(block.minimumSizeHint().width(), self.narrow)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
