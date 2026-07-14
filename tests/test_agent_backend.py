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
        # Needs no key or process, so it is the guaranteed default backend.
        self.assertTrue(agent.EchoBackend().available())

    def test_send_is_deterministic_and_safe(self):
        backend = agent.EchoBackend()
        first = backend.send("hello", "scene", [])
        second = backend.send("hello", "scene", [])
        self.assertEqual(first, second)
        self.assertIsInstance(first, agent.BackendResponse)
        self.assertEqual(first.commands, [])  # no drawing: safe no-op
        self.assertIn("hello", first.text)


class RegistryTC(unittest.TestCase):
    def test_echo_is_always_available(self):
        # EchoBackend registers on import, so the selector always has an
        # entry.
        names = [b.name for b in agent.available_backends()]
        self.assertIn(agent.EchoBackend().name, names)

    def test_get_backend_by_name(self):
        backend = agent.get_backend(agent.EchoBackend().name)
        self.assertIsNotNone(backend)
        self.assertEqual(backend.name, agent.EchoBackend().name)

    def test_register_replaces_same_name(self):
        # Re-registering a name swaps the instance, so a re-import cannot grow
        # the registry.
        before = len(agent.all_backends())

        class Echo2(agent.EchoBackend):
            pass

        replacement = Echo2()
        try:
            agent.register(replacement)
            self.assertEqual(len(agent.all_backends()), before)
            self.assertIs(agent.get_backend(agent.EchoBackend().name),
                          replacement)
        finally:
            # Restore the default for other tests.
            agent.register(agent.EchoBackend())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
