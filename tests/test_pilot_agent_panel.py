# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for when the pilot agent console opens.
"""

import unittest

import solvcon

try:
    from solvcon.pilot.agent import _agent_gui
    from solvcon.pilot.base import _gui
except ImportError:
    _agent_gui = _gui = None


@unittest.skipIf(not solvcon.HAS_PILOT, "pilot is not built")
class AgentAutoOpenTC(unittest.TestCase):
    """The console opens with the canvas commands the table names."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.agent = _gui.controller.agent
        self.model = self.mgr.menu_model
        # The controller is a process-wide singleton, so an earlier test may
        # leave the console open; start every case from the hidden state.
        self.model.action("panel.agent_console").setChecked(False)

    def tearDown(self):
        self.mgr.mdiArea.closeAllSubWindows()
        self.model.action("panel.agent_console").setChecked(False)

    def test_every_table_id_is_a_registered_command(self):
        # bind_auto_open skips a stale id, so nothing else reports one that
        # no feature registers any more.
        missing = [command for command in _agent_gui.AUTO_OPEN_COMMANDS
                   if self.model.action(command) is None]
        self.assertEqual(missing, [])

    def test_new_2d_canvas_opens_the_console(self):
        # trigger() is the single path the menu item and its New shortcut
        # share, so one case covers both.
        self.model.action("canvas.blank_2d").trigger()
        self.assertFalse(self.agent._dock.isHidden())

    def test_the_next_canvas_reopens_a_closed_console(self):
        self.model.action("canvas.blank_2d").trigger()
        self.model.action("panel.agent_console").setChecked(False)
        self.assertTrue(self.agent._dock.isHidden())
        self.model.action("canvas.open_2d").trigger()
        self.assertFalse(self.agent._dock.isHidden())

    def test_a_canvas_opened_outside_a_command_leaves_the_console_alone(self):
        # The table binds menu commands, so a canvas the Python console or the
        # agent itself opens does not drag the console along with it.
        self.mgr.add2DWidget()
        self.assertFalse(self.model.action("panel.agent_console").isChecked())
        dock = self.agent._dock
        self.assertTrue(dock is None or dock.isHidden())


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
