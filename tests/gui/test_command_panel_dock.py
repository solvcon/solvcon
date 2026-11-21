# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the command panel docked in the pilot: Python blocks, notes,
agent turns, Stop, undoing a turn, and replaying it as Python.
"""

import threading
import unittest
from unittest import mock

import solvcon
from solvcon import testing

try:
    from solvcon import pilot
    from solvcon import agent
    from solvcon.pilot.agent import _agent_turn
    from solvcon.pilot.panel import _command
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QLabel, QToolButton
except ImportError:
    pilot = None


class _ScriptBackend:
    """Return each canned reply in turn, then say the work is done;
    optionally hold the first reply until released."""

    name = "script (test)"

    def __init__(self, *replies, gate=None):
        self._replies = list(replies)
        self._gate = gate
        self.cancelled = False
        self.calls = 0

    def available(self):
        return True

    def cancel(self):
        self.cancelled = True

    def send(self, prompt, scene_context, tool_surface, history=()):
        self.calls += 1
        if self._gate is not None and self.calls == 1:
            self._gate.wait(5)
        if self._replies:
            return agent.BackendResponse(text="ok",
                                         commands=self._replies.pop(0))
        return agent.BackendResponse(text="done")


TWO_SHAPES = [{"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 10.0},
              {"op": "add_rectangle", "x_min": -5.0, "y_min": -5.0,
               "x_max": 5.0, "y_max": 5.0}]


@testing.requires_live_window
class CommandPanelTC(unittest.TestCase):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.feature = _command.CommandPanel(mgr=self.mgr)
        self.feature.populate_menu()
        self.panel = self.feature.panel
        self.widget = self.mgr.add2DWidget()
        self.world = solvcon.WorldFp64()
        self.widget.updateWorld(self.world)
        # A main window an earlier test showed lets layout move the view
        # mid-turn, which a turn reads as a canvas change.  Later tests may
        # rely on it being shown, so it comes back.
        main = self.mgr.mainWindow
        if main.isVisible():
            main.hide()
            self.addCleanup(main.show)
        QCoreApplication.processEvents()

    def tearDown(self):
        self.feature.stop()
        self.panel.clear()

    def _use(self, backend):
        combo = self.panel._backend_combo
        combo.insertItem(0, backend.name, backend)
        combo.setCurrentIndex(0)

    def _finish(self):
        runner = self.feature._runner
        for _ in range(10):
            if not runner.busy:
                return
            if runner.worker is not None:
                self.assertTrue(runner.worker.wait(5000))
            QCoreApplication.processEvents()
        self.fail("the turn did not finish")

    def _buttons(self, block):
        return {b.text(): b for b in block.findChildren(QToolButton)}

    def test_a_feature_message_lands_as_a_note(self):
        pilot.base.write_message("mesh: 8 cells\n")
        block = self.panel.blocks()[-1]
        self.assertEqual(block.kind, _command.NOTE)
        self.assertEqual(block._input.text(), "mesh: 8 cells")

    def test_python_block_holds_its_output(self):
        block = self.feature.run_python("print('hello')\n6 * 7")
        texts = [label.text() for label in block.findChildren(QLabel)]
        self.assertIn("hello\n42", texts)

    def test_output_that_looks_like_markup_shows_as_written(self):
        block = self.feature.run_python("print('<b>bold</b>')")
        [label] = [label for label in block.findChildren(QLabel)
                   if label.text() == "<b>bold</b>"]
        self.assertEqual(label.textFormat(), Qt.PlainText)

    def test_a_dock_tabbed_behind_another_stays_open(self):
        dock = self.feature._dock
        dock.show()
        self.feature._action.setChecked(True)
        # What Qt says when another dock's tab comes to the front.
        dock.visibilityChanged.emit(False)
        self.assertTrue(self.feature._action.isChecked())
        self.assertFalse(dock.isHidden())

    def test_a_backend_failure_shows_as_an_error(self):
        class Broken(_ScriptBackend):
            def send(self, prompt, scene_context, tool_surface, history=()):
                raise OSError("no route")

        self._use(Broken())
        block = self.feature.ask_agent("draw")
        self._finish()
        [label] = block.findChildren(QLabel, "error")
        self.assertIn("no route", label.text())
        self.assertTrue(block._status.text().startswith("error"))

    def test_agent_turn_lists_every_command(self):
        self._use(_ScriptBackend(TWO_SHAPES))
        block = self.feature.ask_agent("draw two shapes")
        self._finish()
        self.assertEqual(self.world.nshape, 2)
        rows = [label.text() for label in block.findChildren(QLabel)
                if label.text().startswith(_command._CHECK)]
        self.assertEqual(len(rows), 2)

    def test_undo_turn_reverts_the_whole_turn(self):
        self._use(_ScriptBackend(TWO_SHAPES))
        block = self.feature.ask_agent("draw two shapes")
        self._finish()
        self.assertEqual(self.world.nshape, 2)
        self._buttons(block)["Undo turn"].click()
        self.assertEqual(self.world.nshape, 0)

    def test_no_undo_after_the_user_drew_on(self):
        self._use(_ScriptBackend(TWO_SHAPES))
        block = self.feature.ask_agent("draw two shapes")
        self._finish()
        self.world.add_circle(5.0, 5.0, 1.0)
        undo = self._buttons(block)["Undo turn"]
        undo.click()
        self.assertEqual(self.world.nshape, 3)
        self.assertFalse(undo.isEnabled())

    def test_no_second_turn_while_one_runs(self):
        gate = threading.Event()
        self._use(_ScriptBackend(TWO_SHAPES, gate=gate))
        first = self.feature.ask_agent("draw two shapes")
        self.assertIsNone(self.feature.ask_agent("and more"))
        self.assertEqual(self.panel.blocks(), [first])
        gate.set()
        self._finish()
        self.assertTrue(first._status.text().startswith("ok"))

    def test_a_turn_that_cannot_start_ends_with_an_error(self):
        self._use(_ScriptBackend(TWO_SHAPES))
        session = self.feature._runner.session
        with mock.patch.object(session, "bind_world",
                               side_effect=RuntimeError("no world")):
            block = self.feature.ask_agent("draw")
        self.assertFalse(self.feature._runner.busy)
        self.assertTrue(block._status.text().startswith("error"))
        self.assertEqual(self.panel.run.text(), "Run")
        self.assertTrue(any("no world" in label.text()
                            for label in block.findChildren(QLabel)))
        # The next request must not carry a failure without its prompt.
        self.assertFalse(any(turn.failed for turn in session.transcript))

    def test_a_broken_scene_ends_the_turn_with_an_error(self):
        boom = RuntimeError("boom")
        # The scene is composed for a step, then again to check its reply.
        for name, scenes in (("composing the first step", [boom]),
                             ("checking the reply", ["scene", boom]),
                             ("composing the next step",
                              ["scene", "scene", boom])):
            with self.subTest(name), mock.patch.object(
                    _agent_turn._agent_control, "pilot_scene_context",
                    side_effect=scenes):
                self._use(_ScriptBackend(TWO_SHAPES))
                block = self.feature.ask_agent("draw")
                self._finish()
                self.assertFalse(self.feature._runner.busy)
                self.assertTrue(block._status.text().startswith("error"))
                [label] = [label for label
                           in block.findChildren(QLabel, "error")
                           if "boom" in label.text()]
                self.assertEqual(self.panel.run.text(), "Run")

    def test_stop_drops_a_late_reply(self):
        gate = threading.Event()
        backend = _ScriptBackend(TWO_SHAPES, gate=gate)
        self._use(backend)
        block = self.feature.ask_agent("draw two shapes")
        self.feature.stop()
        self.assertTrue(backend.cancelled)
        gate.set()
        self._finish()
        self.assertEqual(self.world.nshape, 0)
        self.assertTrue(block._status.text().startswith("stopped"))
        self.assertEqual(backend.calls, 1)

    def test_canvas_change_during_a_turn_applies_nothing(self):
        gate = threading.Event()
        self._use(_ScriptBackend(TWO_SHAPES, gate=gate))
        block = self.feature.ask_agent("draw two shapes")
        self.world.add_circle(50.0, 50.0, 1.0)
        gate.set()
        self._finish()
        self.assertEqual(self.world.nshape, 1)
        self.assertTrue(block._status.text().startswith("stopped"))

    def test_as_python_replays_the_turn(self):
        self._use(_ScriptBackend(TWO_SHAPES))
        block = self.feature.ask_agent("draw two shapes")
        self._finish()
        self._buttons(block)["As Python"].click()
        composer = self.panel.composer
        self.assertEqual(composer.mode, _command.PYTHON)
        self.feature.run_python(composer.toPlainText())
        self.assertEqual(self.world.nshape, 4)


@testing.requires_live_window
class ConsoleNoteTC(unittest.TestCase):

    def test_the_note_leads_to_the_full_list(self):
        from solvcon.pilot.base import _gui
        _gui.controller.build()
        with mock.patch.object(_gui._gui_common, "write_message") as write:
            _gui.controller._seed_console_namespace()
        write.assert_called_once()
        note = write.call_args.args[0]
        self.assertEqual(len(note.splitlines()), 1)
        self.assertIn("help_pilot()", note)

        block = _gui.controller.command_panel.run_python("help_pilot()")
        [label] = [label for label in block.findChildren(QLabel)
                   if "Handles in scope:" in label.text()]
        self.assertIn("show_mesh(m)", label.text())
        # The whole list, up to its last entry, help_pilot().
        self.assertIn("capture(target)", label.text())
        self.assertIn("help_pilot()", label.text().splitlines()[-1])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
