# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the console panel's pieces that need no live window: the
Enter rule, the cell runner, and the composer's keys.
"""

import gc
import types
import unittest
from unittest import mock

from solvcon import apputil, testing

try:
    from solvcon import pilot
    from solvcon.pilot.panel import _console
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None


@testing.requires_pilot
class EnterRuleTC(unittest.TestCase):

    def test_open_block_asks_for_more(self):
        self.assertFalse(_console.python_is_complete(
            "for i in range(3):"))
        self.assertFalse(_console.python_is_complete(
            "for i in range(3):\n    print(i)"))

    def test_blank_last_line_closes_the_block(self):
        self.assertTrue(_console.python_is_complete(
            "for i in range(3):\n    print(i)\n"))

    def test_a_blank_line_does_not_close_a_string_or_a_bracket(self):
        for source in ('s = """\n', 'f(\n', 'f(1,\n    ', 's = """\n    '):
            with self.subTest(source=source):
                self.assertFalse(_console.python_is_complete(source))

    def test_broken_source_runs_to_show_the_error(self):
        self.assertTrue(_console.python_is_complete("x = )"))


@testing.requires_pilot
class RunCellTC(unittest.TestCase):

    def _text(self, runs, is_error=None):
        return "".join(t for e, t in runs if is_error is None or e == is_error)

    def test_dedented_statement_after_a_block_runs(self):
        # Pushed line by line into the interactive console this is a
        # SyntaxError; as a cell it runs like a file.
        runs = _console.run_cell(
            "for i in range(2):\n    print(i)\nprint('done')")
        self.assertEqual(self._text(runs), "0\n1\ndone\n")

    def test_stdout_and_stderr_keep_their_order(self):
        runs = _console.run_cell(
            "import sys\nprint('a')\nprint('b', file=sys.stderr)\n"
            "print('c')")
        self.assertEqual(runs, [(False, "a\n"), (True, "b\n"),
                                (False, "c\n")])

    def test_stderr_output_alone_is_not_an_exception(self):
        runs = _console.run_cell(
            "import sys\nprint('careful', file=sys.stderr)")
        self.assertIn("careful", self._text(runs, True))
        self.assertFalse(runs.raised)
        self.assertTrue(_console.run_cell("1 / 0").raised)
        self.assertTrue(_console.run_cell("1 +").raised)

    def test_state_persists_in_the_console_namespace(self):
        _console.run_cell("_cp_value = 5")
        self.assertEqual(
            self._text(_console.run_cell("_cp_value + 1")), "6\n")

    def test_input_reads_eof(self):
        err = self._text(_console.run_cell("input()"), True)
        self.assertIn("EOFError", err)

    def test_only_a_trailing_expression_is_shown(self):
        import builtins
        runs = _console.run_cell("21 + 21")
        self.assertEqual(self._text(runs), "42\n")
        self.assertEqual(builtins._, 42)
        # A statement shows nothing.
        self.assertEqual(self._text(_console.run_cell("_cp_x = 1")), "")

    def test_an_error_shows_the_cell_and_not_the_host(self):
        # Ubuntu's Python installs its own hook, which must not take the
        # report away from the transcript.
        with mock.patch("sys.excepthook", lambda *args: None):
            err = self._text(
                _console.run_cell('raise ValueError("boom")'), True)
        self.assertIn("ValueError: boom", err)
        self.assertIn('File "<console>"', err)
        self.assertNotIn("_console.py", err)
        err = self._text(_console.run_cell("def bad(:"), True)
        self.assertIn("SyntaxError", err)
        self.assertIn("bad(", err)

    def test_a_cell_sees_the_refreshed_namespace(self):
        appenv = apputil.get_current_appenv()

        def refresh(namespace):
            namespace["_cp_live"] = 7
        with mock.patch.object(appenv, "namespace_refreshers", [refresh]):
            runs = _console.run_cell("_cp_live")
        self.assertEqual(self._text(runs), "7\n")

    def test_exit_is_reported_and_the_pilot_lives_on(self):
        runs = _console.run_cell("raise SystemExit(2)")
        self.assertTrue(runs.raised)
        self.assertIn("use the window controls", self._text(runs, True))

    def test_a_failing_cell_gives_the_streams_back(self):
        import sys
        streams = (sys.stdin, sys.stdout, sys.stderr)
        for source in ("raise SystemExit(2)", "x = )", "1 / 0"):
            with self.subTest(source=source):
                _console.run_cell(source)
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

        panel = _console.ConsolePanelWidget()
        kept = panel.add_block(_console.PYTHON, "x = 1")
        gone = panel.add_block(_console.PYTHON, "y = 2")
        before = wrappers()
        self.assertEqual(panel.blocks(), [kept, gone])
        self.assertEqual(wrappers(), before)
        panel.clear(keep=kept)
        self.assertEqual(panel.blocks(), [kept])

    def test_the_transcript_drops_its_oldest_blocks(self):
        panel = _console.ConsolePanelWidget()
        with mock.patch.object(_console, "KEEP_BLOCKS", 2):
            running = panel.add_block(_console.AGENT, "draw")
            panel.add_block(_console.PYTHON, "x = 1")
            # The block of a running turn stays, however old.
            second = panel.add_block(_console.PYTHON, "y = 2", keep=running)
            self.assertEqual(panel.blocks(), [running, second])
            third = panel.add_block(_console.PYTHON, "z = 3")
            self.assertEqual(panel.blocks(), [second, third])


@testing.requires_pilot
class ComposerTC(unittest.TestCase):

    def setUp(self):
        pilot.RManager.instance.setUp()
        self.composer = _console.ConsoleComposer()
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
        self.assertEqual(self.composer.mode, _console.AGENT)
        self._key(QtCore.Qt.Key_I, "", QtCore.Qt.ControlModifier)
        self.assertEqual(self.composer.mode, _console.PYTHON)

    def test_backspace_on_empty_input_switches_back(self):
        self.composer.set_mode(_console.AGENT)
        self._key(QtCore.Qt.Key_Backspace)
        self.assertEqual(self.composer.mode, _console.PYTHON)

    def test_esc_stops_a_turn_only_in_agent_mode(self):
        stops = []
        self.composer.stop_requested.connect(lambda: stops.append(True))
        self._key(QtCore.Qt.Key_Escape)
        self.assertEqual(stops, [])
        self.composer.set_mode(_console.AGENT)
        self._key(QtCore.Qt.Key_Escape)
        self.assertEqual(stops, [True])

    def test_tab_leaves_an_agent_prompt_alone(self):
        self.composer.set_mode(_console.AGENT)
        self.composer.set_text("draw a circle")
        self._key(QtCore.Qt.Key_Tab, "\t")
        self.assertEqual(self.composer.toPlainText(), "draw a circle")

    def _history_walks(self):
        steps = []
        self.composer.navigate.connect(steps.append)
        return steps

    def test_arrows_flagged_as_keypad_keys_walk_the_history(self):
        # What macOS sends for the plain arrow keys.
        steps = self._history_walks()
        self._key(QtCore.Qt.Key_Up, mod=QtCore.Qt.KeypadModifier)
        self._key(QtCore.Qt.Key_Down, mod=QtCore.Qt.KeypadModifier)
        self.assertEqual(steps, [-1, 1])

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

    def test_tab_finishes_the_only_match(self):
        apputil.get_current_appenv().seed(composer_only_match=1)
        self.composer.set_text("x = composer_only_m")
        self._key(QtCore.Qt.Key_Tab, "\t")
        self.assertEqual(self.composer.toPlainText(),
                         "x = composer_only_match")

    def test_tab_on_a_blank_line_indents(self):
        self.composer.set_text("if x:\n")
        self._key(QtCore.Qt.Key_Tab, "\t")
        self.assertEqual(self.composer.toPlainText(), "if x:\n    ")

    def test_completions_see_the_current_viewer(self):
        # The namespace tracks the focused viewer only through its refreshers,
        # so a completion made before the next command needs them to run too.
        appenv = apputil.get_current_appenv()
        probe = types.SimpleNamespace(composer_fresh=1)

        def refresh(namespace):
            namespace["composer_viewer"] = probe
        with mock.patch.object(appenv, "namespace_refreshers", [refresh]):
            self.assertEqual(
                _console.python_completions("composer_viewer.composer"),
                ["composer_fresh"])

    def test_an_object_that_fails_lookup_offers_nothing(self):
        class Broken:
            def __dir__(self):
                raise RuntimeError("no listing")
        apputil.get_current_appenv().seed(composer_broken=Broken())
        self.assertEqual(
            _console.python_completions("composer_broken."), [])


@testing.requires_pilot
class BlockWidthTC(unittest.TestCase):
    """A run with no space to wrap at must not widen the transcript, or
    every block's status slides out of the dock."""

    def setUp(self):
        pilot.RManager.instance.setUp()
        self.panel = _console.ConsolePanelWidget()
        self.addCleanup(self.panel.deleteLater)
        self.narrow = self.panel.add_block(
            _console.PYTHON, "x = 1").minimumSizeHint().width()

    def test_long_output_keeps_the_block_narrow(self):
        block = self.panel.add_block(_console.PYTHON, "x = 1")
        block.add_output("x" * 1000)
        self.assertEqual(block.minimumSizeHint().width(), self.narrow)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
