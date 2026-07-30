# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon
from pilot_ci import SKIP_PILOT_WIDGETS

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.base import _gui_common
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None
    QtGui = None

# QtTest is its own PySide6 extension, and the macOS CI job rewrites the Qt
# rpath of QtCore, QtGui, and QtWidgets only, so importing it there fails.
# Keeping it out of the block above stops that from unbinding pilot for the
# whole module, which would fail every test in the file rather than skipping
# the two that need live key delivery.
try:
    from PySide6.QtTest import QTest
except ImportError:
    QTest = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))

# KeyMod::Primary and Key::Z ordinals from keymap.hpp, passed to
# shortcut_conflicts_rebinding; a static_assert in wrap_pilot.cpp pins them.
_KEYMOD_PRIMARY = 1
_KEY_Z = 17

_PANEL_CHORDS = (
    ("panel.agent_console", ["Ctrl+Shift+A"]),
    ("panel.inspector", ["Ctrl+Shift+I"]),
    ("panel.painter", ["Ctrl+Shift+P"]),
)

_DRAW_TOOL_KEYS = (
    ("select", "V"),
    ("line", "L"),
    ("triangle", "T"),
    ("rectangle", "R"),
    ("ellipse", "E"),
    ("circle", "C"),
)

if QtGui is not None:
    _STANDARD_IDS = {
        "edit.undo": QtGui.QKeySequence.StandardKey.Undo,
        "edit.redo": QtGui.QKeySequence.StandardKey.Redo,
        "file.exit": QtGui.QKeySequence.StandardKey.Quit,
        "canvas.blank_2d": QtGui.QKeySequence.StandardKey.New,
    }


def _can_take_keyboard(widget):
    """Whether the platform will hand ``widget`` the keyboard.

    A run whose terminal stays frontmost never gets an active window, and
    activateWindow cannot argue with that. Probe it so a test needing real key
    delivery skips instead of failing for a reason unrelated to the code.
    """
    widget.window().activateWindow()
    widget.window().raise_()
    widget.setFocus()
    QtWidgets.QApplication.processEvents()
    return widget.hasFocus() and widget.isActiveWindow()


def _live_sequences(action):
    return [s.toString(QtGui.QKeySequence.PortableText)
            for s in action.shortcuts()]


def _portable_standard_sequences(standard_key):
    return [s.toString(QtGui.QKeySequence.PortableText)
            for s in QtGui.QKeySequence.keyBindings(standard_key)]


@unittest.skipIf(not solvcon.HAS_PILOT, "GUI is not available")
class ShortcutResolutionTC(unittest.TestCase):
    """The roof resolves a command id to portable values the Python layer can
    apply. Runs under QT_QPA_PLATFORM=offscreen on CI when pilot is built."""

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
        self.assertEqual(r["sequences"], _portable_standard_sequences(
            QtGui.QKeySequence.StandardKey.Undo))

    def test_camera_reset_resolves_to_a_curated_chord(self):
        r = self.mgr.resolve_shortcut("camera.reset")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertFalse(r["standard"])
        self.assertEqual(r["context"], "widget")
        self.assertEqual(r["role"], "none")
        self.assertEqual(r["sequences"], ["Esc"])

    def test_console_resolves_to_the_grave_toggle(self):
        r = self.mgr.resolve_shortcut("window.console")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertFalse(r["standard"])
        self.assertEqual(r["context"], "window")
        # Physical Ctrl+` on every platform, matching VSCode's terminal
        # toggle. Qt reaches the macOS Control key through Meta, so the
        # portable text spells it "Meta+`" there.
        if self.mgr.shortcut_platform == "mac":
            self.assertEqual(r["sequences"], ["Meta+`"])
        else:
            self.assertEqual(r["sequences"], ["Ctrl+`"])

    def test_panel_chords_resolve(self):
        for oid, sequences in _PANEL_CHORDS:
            with self.subTest(oid=oid):
                r = self.mgr.resolve_shortcut(oid)
                self.assertTrue(r["known"])
                self.assertTrue(r["bound"])
                self.assertEqual(r["sequences"], sequences)

    def test_draw_tools_resolve_to_unmodified_letters(self):
        for tool, key in _DRAW_TOOL_KEYS:
            with self.subTest(tool=tool):
                r = self.mgr.resolve_shortcut("draw.tool." + tool)
                self.assertTrue(r["known"])
                self.assertTrue(r["bound"])
                self.assertFalse(r["standard"])
                self.assertEqual(r["sequences"], [key])
                self.assertEqual(r["context"], "widget")

    def test_new_2d_canvas_resolves_to_new(self):
        r = self.mgr.resolve_shortcut("canvas.blank_2d")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "New")
        self.assertEqual(r["sequences"], _portable_standard_sequences(
            QtGui.QKeySequence.StandardKey.New))

    def test_exit_carries_quit_and_platform_role(self):
        r = self.mgr.resolve_shortcut("file.exit")
        self.assertTrue(r["known"])
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "Quit")
        self.assertEqual(r["context"], "application")
        self.assertEqual(r["sequences"], _portable_standard_sequences(
            QtGui.QKeySequence.StandardKey.Quit))
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

    def test_every_vocabulary_command_resolves(self):
        # Standard keys may be unbound on a platform (Quit has no chord on
        # Windows); curated phrasebook rows must still carry a sequence.
        all_bindings = self.mgr.resolve_all_shortcuts()
        for oid, entry in all_bindings.items():
            with self.subTest(oid=oid):
                self.assertTrue(entry["bound"], oid)
                if oid in _STANDARD_IDS:
                    self.assertEqual(
                        entry["sequences"],
                        _portable_standard_sequences(_STANDARD_IDS[oid]),
                        oid)
                else:
                    self.assertTrue(entry["sequences"], oid)

    def test_capabilities_match_the_running_platform(self):
        caps = self.mgr.shortcut_capabilities
        self.assertEqual(caps["moves_items_to_application_menu"],
                         self.mgr.shortcut_platform == "mac")
        self.assertGreater(caps["reserved_count"], 0)

    def test_primary_modifier_native_text_follows_platform(self):
        # Qt PortableText spells the command modifier as Ctrl everywhere;
        # NativeText carries the Command glyph on macOS.
        r = self.mgr.resolve_shortcut("panel.agent_console")
        self.assertEqual(r["sequences"], ["Ctrl+Shift+A"])
        native = QtGui.QKeySequence(
            r["sequences"][0], QtGui.QKeySequence.PortableText)
        native_text = native.toString(QtGui.QKeySequence.NativeText)
        if self.mgr.shortcut_platform == "mac":
            self.assertIn(chr(0x2318), native_text)
        else:
            self.assertIn("Ctrl", native_text)

    def test_resolved_checker_reports_curated_undo_clash(self):
        # Rebind Console to Primary+Z; the resolved checker must see Undo.
        conflicts = self.mgr.shortcut_conflicts_rebinding(
            "window.console", _KEYMOD_PRIMARY, _KEY_Z)
        pairs = {frozenset(pair) for pair in conflicts}
        self.assertIn(frozenset({"edit.undo", "window.console"}), pairs)


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
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

    def test_draw_tool_actions_carry_the_resolved_bindings(self):
        for tool, _key in _DRAW_TOOL_KEYS:
            with self.subTest(tool=tool):
                self._assert_action_matches_resolved(
                    "draw.tool." + tool, QtCore.Qt.WidgetShortcut)

    def test_exit_action_carries_quit_and_platform_role(self):
        self._assert_action_matches_resolved(
            "file.exit", QtCore.Qt.ApplicationShortcut)
        action = self.model.action("file.exit")
        if self.mgr.shortcut_platform == "mac":
            self.assertEqual(action.menuRole(), QtGui.QAction.QuitRole)
        else:
            self.assertEqual(action.menuRole(), QtGui.QAction.NoRole)


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
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


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS
                 or not solvcon.HAS_PILOT or QTest is None,
                 "live key delivery needs a real window surface and QtTest")
class DrawToolShortcutTC(unittest.TestCase):
    """The draw-tool letters reach a focused canvas and nothing else."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.mgr.add2DWidget()
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        self.target = self.sub.widget()
        self.mgr.setDrawTool("select")
        QtWidgets.QApplication.processEvents()

    def _type(self, widget, key):
        if not _can_take_keyboard(widget):
            self.skipTest("the platform did not grant the keyboard")
        QTest.keyClick(widget, key)
        QtWidgets.QApplication.processEvents()

    def test_canvas_carries_every_draw_tool_action(self):
        names = {a.objectName() for a in self.target.actions()}
        for tool, _key in _DRAW_TOOL_KEYS:
            self.assertIn("draw.tool." + tool, names)

    def test_letter_on_a_focused_canvas_picks_its_tool(self):
        for tool, key in _DRAW_TOOL_KEYS:
            with self.subTest(tool=tool):
                # Start from a tool the letter is not for, so a key that never
                # arrived cannot pass by leaving the choice as it was.
                self.mgr.setDrawTool(
                    "select" if "circle" == tool else "circle")
                self._type(self.target, getattr(QtCore.Qt, "Key_" + key))
                self.assertEqual(self.mgr.drawTool, tool)

    def test_letter_typed_in_a_text_field_stays_in_the_field(self):
        edit = QtWidgets.QLineEdit(self.mgr.mainWindow)
        edit.show()
        try:
            self.mgr.setDrawTool("line")
            self._type(edit, QtCore.Qt.Key_R)
            self.assertEqual(self.mgr.drawTool, "line")
            self.assertEqual(edit.text(), "r")
        finally:
            edit.setParent(None)
            edit.deleteLater()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
