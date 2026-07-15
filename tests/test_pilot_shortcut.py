# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _gui
    from solvcon.pilot import _gui_common
    from PySide6 import QtCore, QtGui
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)

_PANEL_CHORDS = (
    ("panel.agent_console", ["Ctrl+Shift+A"]),
    ("panel.inspector", ["Ctrl+Shift+I"]),
    ("panel.painter", ["Ctrl+Shift+P"]),
)


def _live_sequences(action):
    return [s.toString(QtGui.QKeySequence.PortableText)
            for s in action.shortcuts()]


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ShortcutResolutionTC(unittest.TestCase):
    """The roof resolves a command id to portable values the Python layer can
    apply."""

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def test_platform_is_recognized(self):
        self.assertIn(self.mgr.shortcut_platform, ("linux", "mac", "windows"))

    def test_undo_resolves_to_its_standard_key(self):
        r = self.mgr.resolve_shortcut("edit.undo")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "Undo")
        self.assertEqual(r["context"], "window")
        self.assertTrue(r["sequences"])

    def test_camera_reset_resolves_to_a_curated_chord(self):
        r = self.mgr.resolve_shortcut("camera.reset")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertFalse(r["standard"])
        self.assertEqual(r["context"], "widget")
        self.assertEqual(r["role"], "none")
        self.assertEqual(r["sequences"], ["Esc"])

    def test_console_resolves_to_primary_grave(self):
        r = self.mgr.resolve_shortcut("window.console")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertFalse(r["standard"])
        self.assertEqual(r["context"], "window")
        self.assertEqual(r["sequences"], ["Ctrl+`"])

    def test_panel_chords_resolve(self):
        for oid, sequences in _PANEL_CHORDS:
            with self.subTest(oid=oid):
                r = self.mgr.resolve_shortcut(oid)
                self.assertTrue(r["known"])
                self.assertTrue(r["bound"])
                self.assertEqual(r["sequences"], sequences)

    def test_new_2d_canvas_resolves_to_new(self):
        r = self.mgr.resolve_shortcut("canvas.blank_2d")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "New")

    def test_exit_carries_quit_and_platform_role(self):
        r = self.mgr.resolve_shortcut("file.exit")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "Quit")
        self.assertEqual(r["context"], "application")
        if self.mgr.shortcut_platform == "mac":
            self.assertEqual(r["role"], "quit")
        else:
            self.assertEqual(r["role"], "none")

    def test_unknown_command_is_unbound(self):
        r = self.mgr.resolve_shortcut("no.such.command")
        self.assertFalse(r["known"])
        self.assertFalse(r["bound"])
        self.assertEqual(r["sequences"], [])

    def test_resolved_bindings_do_not_collide(self):
        self.assertEqual(self.mgr.shortcut_conflicts(), [])


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ShortcutTC(unittest.TestCase):
    """Live QAction bindings for the commands routed through the roof."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.model = self.mgr.menu_model

    def _assert_action_matches_resolved(self, object_name, qt_context):
        action = self.model.action(object_name)
        self.assertIsNotNone(action)
        resolved = self.mgr.resolve_shortcut(object_name)
        self.assertTrue(resolved["known"])
        self.assertTrue(resolved["bound"])
        self.assertEqual(_live_sequences(action), resolved["sequences"])
        self.assertEqual(action.shortcutContext(), qt_context)

    def test_undo_action_carries_the_resolved_binding(self):
        self._assert_action_matches_resolved(
            "edit.undo", QtCore.Qt.WindowShortcut)

    def test_redo_action_carries_the_resolved_binding(self):
        self._assert_action_matches_resolved(
            "edit.redo", QtCore.Qt.WindowShortcut)

    def test_camera_reset_action_carries_the_resolved_binding(self):
        self._assert_action_matches_resolved(
            "camera.reset", QtCore.Qt.WidgetShortcut)

    def test_console_action_carries_the_resolved_binding(self):
        self._assert_action_matches_resolved(
            "window.console", QtCore.Qt.WindowShortcut)

    def test_panel_actions_carry_the_resolved_bindings(self):
        for oid, _ in _PANEL_CHORDS:
            with self.subTest(oid=oid):
                self._assert_action_matches_resolved(
                    oid, QtCore.Qt.WindowShortcut)

    def test_new_2d_canvas_action_carries_the_resolved_binding(self):
        self._assert_action_matches_resolved(
            "canvas.blank_2d", QtCore.Qt.WindowShortcut)

    def test_exit_action_carries_quit_and_platform_role(self):
        self._assert_action_matches_resolved(
            "file.exit", QtCore.Qt.ApplicationShortcut)
        action = self.model.action("file.exit")
        if self.mgr.shortcut_platform == "mac":
            self.assertEqual(action.menuRole(), QtGui.QAction.QuitRole)
        else:
            self.assertEqual(action.menuRole(), QtGui.QAction.NoRole)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ApplyShortcutHelperTC(unittest.TestCase):
    """apply_shortcut installs a binding by objectName without hand wiring."""

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def test_applies_a_known_command_to_a_fresh_action(self):
        act = QtGui.QAction("Console", self.mgr.mainWindow)
        act.setObjectName("window.console")
        _gui_common.apply_shortcut(act, mgr=self.mgr)
        resolved = self.mgr.resolve_shortcut("window.console")
        self.assertEqual(_live_sequences(act), resolved["sequences"])
        self.assertEqual(act.shortcutContext(), QtCore.Qt.WindowShortcut)

    def test_unknown_id_is_a_noop(self):
        act = QtGui.QAction("Scratch", self.mgr.mainWindow)
        act.setObjectName("no.such.command")
        act.setShortcut("Ctrl+X")
        act.setShortcutContext(QtCore.Qt.ApplicationShortcut)
        act.setMenuRole(QtGui.QAction.AboutRole)
        _gui_common.apply_shortcut(act, mgr=self.mgr)
        self.assertEqual(act.shortcut().toString(
            QtGui.QKeySequence.PortableText), "Ctrl+X")
        self.assertEqual(act.shortcutContext(),
                         QtCore.Qt.ApplicationShortcut)
        self.assertEqual(act.menuRole(), QtGui.QAction.AboutRole)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
