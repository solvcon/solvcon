# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the Agent backend abstraction and registry.

GUI-free: only the pure-Python backend module is imported, never an
``RManager`` or a Qt widget, so these run in CI without a built GUI.
"""

import unittest

from solvcon import agent


class AgentBackendABCTC(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            agent.AgentBackend()

    def test_partial_subclass_cannot_instantiate(self):
        # Missing send() leaves an abstract method, guarding the contract that
        # every concrete backend (Claude, Codex, ...) fills all three.
        class Partial(agent.AgentBackend):
            name = "partial"

            def available(self):
                return True

        with self.assertRaises(TypeError):
            Partial()


class EchoBackendTC(unittest.TestCase):
    def test_available_true_without_config(self):
        # Needs no key or process, so a test can always drive it.
        self.assertTrue(agent.EchoBackend().available())

    def test_send_is_deterministic_and_safe(self):
        backend = agent.EchoBackend()
        first = backend.send("hello", "scene", [])
        second = backend.send("hello", "scene", [])
        self.assertEqual(first, second)
        self.assertIsInstance(first, agent.BackendResponse)
        self.assertEqual(first.commands, [])
        self.assertIn("hello", first.text)


class RegistryTC(unittest.TestCase):
    def test_echo_is_not_offered(self):
        # The offline double stays a test tool, out of the user's selector.
        names = [b.name for b in agent.all_backends()]
        self.assertNotIn(agent.EchoBackend().name, names)

    def test_claude_cli_is_the_first_entry(self):
        # Registration order is selector order, so the Claude CLI is what a
        # selector starts on.
        self.assertEqual(agent.all_backends()[0].name,
                         agent.ClaudeCliBackend().name)

    def test_get_backend_by_name(self):
        name = agent.ClaudeCliBackend().name
        backend = agent.get_backend(name)
        self.assertIsNotNone(backend)
        self.assertEqual(backend.name, name)

    def test_register_replaces_same_name(self):
        # Re-registering a name swaps the instance, so a re-import cannot grow
        # the registry.
        before = len(agent.all_backends())

        class Claude2(agent.ClaudeCliBackend):
            pass

        replacement = Claude2()
        try:
            agent.register(replacement)
            self.assertEqual(len(agent.all_backends()), before)
            self.assertIs(agent.get_backend(replacement.name), replacement)
        finally:
            # Restore the default for other tests.
            agent.register(agent.ClaudeCliBackend())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
