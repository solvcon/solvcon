# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class ShortcutResolutionTC(unittest.TestCase):
    """The roof resolves a command id to portable values the Python layer can
    apply. No action is routed through it yet; these check the resolution."""

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def test_platform_is_recognized(self):
        self.assertIn(self.mgr.shortcut_platform, ("linux", "mac", "windows"))

    def test_undo_resolves_to_its_standard_key(self):
        r = self.mgr.resolve_shortcut("edit.undo")
        self.assertTrue(r["bound"])
        self.assertTrue(r["standard"])
        self.assertEqual(r["standard_key"], "Undo")
        self.assertEqual(r["context"], "window")
        self.assertTrue(r["sequences"])

    def test_camera_reset_resolves_to_a_curated_chord(self):
        r = self.mgr.resolve_shortcut("camera.reset")
        self.assertTrue(r["bound"])
        self.assertFalse(r["standard"])
        self.assertEqual(r["context"], "widget")
        self.assertEqual(r["role"], "none")
        self.assertEqual(r["sequences"], ["Esc"])

    def test_exit_role_follows_the_platform(self):
        r = self.mgr.resolve_shortcut("file.exit")
        if self.mgr.shortcut_platform == "mac":
            # macOS carries the Quit application-menu role with no key yet.
            self.assertFalse(r["bound"])
            self.assertEqual(r["role"], "quit")
            self.assertEqual(r["context"], "application")
        else:
            # No Exit entry off macOS until a later step binds it.
            self.assertFalse(r["bound"])
            self.assertEqual(r["role"], "none")

    def test_unknown_command_is_unbound(self):
        r = self.mgr.resolve_shortcut("no.such.command")
        self.assertFalse(r["bound"])
        self.assertEqual(r["sequences"], [])

    def test_resolved_bindings_do_not_collide(self):
        self.assertEqual(self.mgr.shortcut_conflicts(), [])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
