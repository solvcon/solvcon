# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the headless AgentSession core (GUI-free); a fake runner and
world cover the orchestration, and the Agent Draw integration cases require
the ``agentdraw`` package."""

import json
import unittest

import solvcon
from solvcon import agent

try:
    from solvcon.pilot import agentdraw  # noqa: F401
    HAS_AGENTDRAW = True
except ImportError:
    HAS_AGENTDRAW = False


class _FakeResult:
    """Minimal Agent Draw result stand-in: the fields the session reads."""

    def __init__(self, op, ok=True, error=None):
        self.op = op
        self.ok = ok
        self.error = error


class _RecordingRunner:
    """Records commands and returns a result for each; ``fail_ops`` names ops
    returned as failed, exercising the failure path without a real executor."""

    def __init__(self, fail_ops=()):
        self.commands = []
        self._fail_ops = set(fail_ops)

    def run(self, command):
        self.commands.append(command)
        op = command.get("op", "?")
        ok = op not in self._fail_ops
        return _FakeResult(op, ok=ok, error=None if ok else "bad command")


class _FakeWorld:
    """World stub with just nshape and describe_state (no concrete shapes)."""

    def __init__(self, types):
        self._types = list(types)

    @property
    def nshape(self):
        return len(self._types)

    def describe_state(self, level="basic"):
        shapes = [{"id": i, "type": t} for i, t in enumerate(self._types)]
        return json.dumps({"shapes": shapes})


class ApplyCommandsTC(unittest.TestCase):
    def test_applies_each_command_and_records_one_turn(self):
        runner = _RecordingRunner()
        session = agent.AgentSession(runner=runner)
        cmds = [{"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0},
                {"op": "add_circle", "cx": 2.0, "cy": 0.0, "r": 1.0}]
        results = session.apply_commands(cmds)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(runner.commands, cmds)
        # One agent turn captures the whole batch and its outcomes.
        self.assertEqual(len(session.transcript), 1)
        turn = session.transcript[0]
        self.assertEqual(turn.role, "agent")
        self.assertEqual(turn.results, results)

    def test_failed_command_is_recorded_not_raised(self):
        runner = _RecordingRunner(fail_ops={"remove_shape"})
        session = agent.AgentSession(runner=runner)
        results = session.apply_commands(
            [{"op": "remove_shape", "shape_id": 99}])
        self.assertFalse(results[0].ok)
        self.assertEqual(session.transcript[0].results[0].error,
                         "bad command")

    def test_runner_exception_is_captured(self):
        class _Boom:
            def run(self, command):
                raise ValueError("boom")

        session = agent.AgentSession(runner=_Boom())
        results = session.apply_commands([{"op": "add_circle"}])
        self.assertFalse(results[0].ok)
        self.assertIn("boom", results[0].error)
        self.assertEqual(results[0].op, "add_circle")

    def test_empty_batch_is_a_noop_without_runner(self):
        # No commands must not build the runner.
        session = agent.AgentSession()
        self.assertEqual(session.apply_commands([]), [])
        self.assertEqual(session.transcript, [])


class _FakeBackend:
    """Backend stub returning a preset response and recording what it saw."""

    def __init__(self, response):
        self._response = response
        self.seen = None

    def send(self, prompt, scene_context, tool_surface):
        self.seen = (prompt, scene_context, tool_surface)
        return self._response


class RunTurnTC(unittest.TestCase):
    def test_records_user_and_agent_turns_and_runs_commands(self):
        cmds = [{"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0}]
        backend = _FakeBackend(
            agent.BackendResponse(text="drawing", commands=cmds))
        runner = _RecordingRunner()
        session = agent.AgentSession(backend=backend, runner=runner)
        turn = session.run_turn("draw a circle")
        self.assertEqual(runner.commands, cmds)
        self.assertEqual([t.role for t in session.transcript],
                         ["user", "agent"])
        self.assertEqual(session.transcript[0].text, "draw a circle")
        self.assertIs(turn, session.transcript[1])
        self.assertEqual(turn.text, "drawing")
        self.assertTrue(turn.results[0].ok)

    def test_no_backend_records_only_the_user_turn(self):
        session = agent.AgentSession()
        self.assertIsNone(session.run_turn("hello"))
        self.assertEqual([t.role for t in session.transcript], ["user"])

    def test_empty_command_batch_builds_no_runner(self):
        backend = _FakeBackend(
            agent.BackendResponse(text="echo: hi", commands=[]))
        # No runner, no agentdraw.
        session = agent.AgentSession(backend=backend)
        turn = session.run_turn("hi")
        self.assertEqual(turn.text, "echo: hi")
        self.assertEqual(turn.results, [])
        # No commands must leave the runner unbuilt (no agentdraw import).
        self.assertIsNone(session._runner)

    def test_backend_exception_is_recorded_as_a_failed_turn(self):
        class _Boom:
            def send(self, *a):
                raise RuntimeError("backend down")

        session = agent.AgentSession(backend=_Boom())
        turn = session.run_turn("draw")
        # The turn is recorded, not propagated, so a headless caller still
        # gets a transcript with the failure.
        self.assertEqual([t.role for t in session.transcript],
                         ["user", "agent"])
        self.assertIn("backend down", turn.text)

    def test_runner_build_failure_yields_one_result_per_command(self):
        class _Session(agent.AgentSession):
            @property
            def runner(self):
                raise ImportError("no agentdraw")

        backend = _FakeBackend(agent.BackendResponse(
            commands=[{"op": "add_circle"}, {"op": "add_square"}]))
        session = _Session(backend=backend)
        turn = session.run_turn("draw two shapes")
        # Results line up with commands even when the runner cannot be built.
        self.assertEqual(len(turn.results), 2)
        self.assertFalse(any(r.ok for r in turn.results))
        self.assertEqual([r.op for r in turn.results],
                         ["add_circle", "add_square"])

    def test_backend_error_is_folded_into_the_reply(self):
        backend = _FakeBackend(agent.BackendResponse(error="claude timed out"))
        session = agent.AgentSession(backend=backend)
        turn = session.run_turn("draw")
        self.assertIn("claude timed out", turn.text)

    def test_bind_world_drops_the_lazy_runner(self):
        session = agent.AgentSession(world=_FakeWorld(["circle"]))
        session._runner = object()  # stand in for a built runner
        session.bind_world(_FakeWorld([]))
        self.assertIsNone(session._runner)

    def test_bind_world_keeps_an_injected_runner(self):
        runner = _RecordingRunner()
        session = agent.AgentSession(runner=runner)
        session.bind_world(_FakeWorld([]))
        self.assertIs(session._runner, runner)


class SceneContextTC(unittest.TestCase):
    def test_no_world(self):
        self.assertEqual(agent.AgentSession().scene_context(),
                         "no active world")

    def test_reports_shape_count_and_types(self):
        session = agent.AgentSession(world=_FakeWorld(["circle", "rectangle"]))
        context = session.scene_context()
        self.assertIn("2 shapes", context)
        self.assertIn("circle", context)
        self.assertIn("rectangle", context)


@unittest.skipUnless(HAS_AGENTDRAW, "requires the agentdraw package")
class AgentDrawIntegrationTC(unittest.TestCase):
    def test_default_runner_mutates_world(self):
        world = solvcon.WorldFp64()
        session = agent.AgentSession(world=world)
        results = session.apply_commands([
            {"op": "add_circle", "cx": 0.0, "cy": 0.0, "r": 1.0},
            {"op": "add_circle", "cx": 3.0, "cy": 0.0, "r": 1.0}])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(world.nshape, 2)

    def test_tool_surface_matches_agentdraw(self):
        self.assertEqual(agent.AgentSession().tool_surface(),
                         agentdraw.tool_definitions())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
