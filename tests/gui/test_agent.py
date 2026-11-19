# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

from solvcon import testing

try:
    from solvcon import pilot
    from solvcon import agent
    from solvcon.pilot.agent import _agent_settings
except ImportError:
    pilot = None


@testing.requires_live_window
class AgentBackendSettingsTC(unittest.TestCase):
    """The backend settings dialog the Command panel's Settings opens."""

    def setUp(self):
        pilot.RManager.instance.setUp()

    def test_backend_settings_reach_the_cli(self):
        # The whole path in one go: the dialog is built from the backend's
        # spec, accepting it stores the picks, and a later turn puts them on
        # the CLI command line.
        backend = agent.ClaudeCliBackend()
        dialog = _agent_settings.AgentBackendSettingsDialog(backend)
        dialog._editors["model"].setCurrentText("opus")
        dialog._editors["effort"].setCurrentText("high")
        dialog.accept()
        argv = backend._build_argv("/usr/bin/claude", "draw", "system")
        self.assertIn("--model=opus", argv)
        self.assertIn("--effort=high", argv)

    def test_codex_backend_settings_reach_the_cli(self):
        backend = agent.CodexCliBackend()
        dialog = _agent_settings.AgentBackendSettingsDialog(backend)
        dialog._editors["model"].setCurrentText("gpt-5.6-sol")
        dialog._editors["effort"].setCurrentText("high")
        dialog.accept()
        argv = backend._build_argv("/usr/bin/codex", "draw", "system")
        self.assertIn("--model=gpt-5.6-sol", argv)
        self.assertIn('--config=model_reasoning_effort="high"', argv)

    def test_openai_http_settings_reach_the_request(self):
        # A free-text knob gets a line edit rather than a combo box, and
        # accepting the dialog is what lands the typed text on the backend.
        backend = agent.OpenAIHttpBackend()
        dialog = _agent_settings.AgentBackendSettingsDialog(backend)
        dialog._editors["base_url"].setText("https://api.example.test/v1")
        dialog._editors["model"].setText("gpt-4o-mini")
        dialog.accept()
        self.assertEqual(backend.base_url, "https://api.example.test/v1")
        self.assertEqual(backend.model, "gpt-4o-mini")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
