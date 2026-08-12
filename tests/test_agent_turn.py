# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the bounded turn loop."""

import unittest

import solvcon
from solvcon import agent


_TOOLS = [{"name": "add_circle", "category": "create",
           "description": "add a circle"}]


class _Runner:
    """Command runner that fails whatever op it was told to fail."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        op = agent.op_of(command)
        if op in self.failing:
            return agent.CommandResult(op, False, error="%s: bad args" % op)
        return agent.CommandResult(op, True, value={"shape_id": 1})

    def tool_definitions(self):
        return _TOOLS

    def commands_by_category(self):
        return {"create": ["add_circle"], "delete": []}


class _ScriptedBackend(agent.AgentBackend):
    """Backend that replays canned replies and records what it was asked.

    A reply is either the text a model would have printed, run through the
    real :class:`ToolCallParser`, or a ready :class:`BackendResponse` for the
    transport outcomes no text can express.  A script that runs short raises,
    which says the test asked for a step it did not plan for.
    """

    name = "scripted (offline)"

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.requests = []

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface, history=()):
        self.requests.append(agent.TurnRequest(
            prompt=prompt, scene_context=scene_context,
            tool_surface=list(tool_surface or ()), history=list(history)))
        reply = self.replies.pop(0)
        if isinstance(reply, agent.BackendResponse):
            return reply
        return agent.ToolCallParser.parse_reply(reply).response(reply)


def _session(replies, runner=None, **kwargs):
    return agent.AgentSession(
        backend=_ScriptedBackend(replies),
        runner=runner if runner is not None else _Runner(), **kwargs)


def _ask(session, turn):
    """One step's reply, composed and sent the way a driver would."""
    return turn.next_request().send_to(session.backend)


def _drain(session, turn):
    """Run ``turn`` to its stop."""
    while True:
        request = turn.next_request()
        if request is None:
            return
        turn.feed(request.send_to(session.backend))


class TurnLoopTC(unittest.TestCase):
    def test_model_fixes_its_own_failed_command_within_budget(self):
        runner = _Runner(failing={"add_blob"})
        session = _session(['[{"op": "add_blob"}]',
                            '[{"op": "add_circle"}]',
                            "[]"], runner=runner)
        agent.run_turn(session, "draw a circle", budget=4)
        self.assertEqual([agent.op_of(c) for c in runner.commands],
                         ["add_blob", "add_circle"])
        # The failed command's error is what the second step was composed
        # against, so the model could see what to fix.
        second = session.backend.requests[1]
        self.assertIn("bad args",
                      agent.format_history(second.history))

    def test_empty_batch_ends_the_turn_as_completion(self):
        session = _session(['[{"op": "add_circle"}]', "[]",
                            '[{"op": "add_circle"}]'])
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.COMPLETED)
        self.assertEqual(len(session.backend.requests), 2)
        self.assertEqual(len(session.backend.replies), 1)

    def test_prose_ends_the_turn_with_the_text_recorded(self):
        session = _session(["I cannot draw that."])
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.PROSE)
        self.assertIn("I cannot draw that.", session.transcript[-1].text)

    def test_malformed_costs_a_step_and_retries_with_the_error_shown(self):
        session = _session(['[{"op": "add_circle",}]',
                            '[{"op": "add_circle"}]', "[]"])
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.COMPLETED)
        self.assertEqual(len(session.backend.requests), 3)
        retry = session.backend.requests[1]
        self.assertIn("malformed", agent.format_history(retry.history))

    def test_transport_outcome_aborts_with_no_retry(self):
        gone = agent.BackendResponse(
            error="claude exit 1", outcome=agent.TransportOutcome.TRANSPORT)
        session = _session([gone, '[{"op": "add_circle"}]'])
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.TRANSPORT)
        self.assertEqual(len(session.backend.requests), 1)
        self.assertIn("claude exit 1", session.transcript[-1].text)
        self.assertTrue(session.transcript[-1].failed)

    def test_an_error_without_an_outcome_does_not_read_as_completion(self):
        # A backend that fills only the older error field leaves the outcome
        # ok and the batch empty, which must not pass for the model saying it
        # has finished.
        session = _session([agent.BackendResponse(error="claude timed out")])
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        self.assertNotEqual(turn.stop_reason, agent.StopReason.COMPLETED)
        self.assertTrue(session.transcript[-1].failed)

    def test_cancelled_and_timeout_map_to_their_own_reasons(self):
        for outcome, reason in (
                (agent.TransportOutcome.TIMEOUT, agent.StopReason.TIMEOUT),
                (agent.TransportOutcome.CANCELLED,
                 agent.StopReason.CANCELLED)):
            session = _session(
                [agent.BackendResponse(error="stopped", outcome=outcome)])
            turn = agent.Turn(session, "draw", budget=4)
            _drain(session, turn)
            self.assertEqual(turn.stop_reason, reason)

    def test_unknown_op_fails_alone_without_killing_its_batch(self):
        # The op the model invented is not caught while parsing, so the good
        # command in the same batch still runs and only the bad one fails.
        world = solvcon.WorldFp64()
        session = agent.AgentSession(
            world=world,
            backend=_ScriptedBackend([
                '[{"op": "add_circle", "cx": 0, "cy": 0, "r": 1},'
                ' {"op": "delete_universe"}]', "[]"]))
        turn = agent.Turn(session, "draw", budget=4)
        _drain(session, turn)
        results = session.transcript[1].results
        self.assertEqual([result.ok for result in results], [True, False])
        self.assertIn("unknown op", results[1].error)
        self.assertEqual(world.nshape, 1)

    def test_budget_exhaustion_is_recorded(self):
        session = _session(['[{"op": "add_circle"}]'] * 4)
        turn = agent.Turn(session, "draw", budget=2)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.BUDGET)
        self.assertEqual(len(session.backend.requests), 2)
        last = session.transcript[-1]
        self.assertEqual(last.role, agent.HistoryFormatter.MARKER_ROLE)
        self.assertIn("step budget", last.text)

    def test_one_shot_budget_records_no_budget_marker(self):
        session = _session(['[{"op": "add_circle"}]'])
        turn = agent.Turn(session, "draw", budget=1)
        _drain(session, turn)
        self.assertEqual(turn.stop_reason, agent.StopReason.BUDGET)
        self.assertEqual([t.role for t in session.transcript],
                         ["user", "agent"])

    def test_stop_between_steps_leaves_a_clean_transcript(self):
        session = _session(['[{"op": "add_circle"}]'] * 2)
        turn = agent.Turn(session, "draw", budget=4)
        turn.feed(_ask(session, turn))
        turn.stop()
        self.assertTrue(turn.done)
        self.assertEqual(turn.stop_reason, agent.StopReason.STOPPED)
        self.assertIsNone(turn.next_request())
        self.assertEqual([t.role for t in session.transcript],
                         ["user", "agent"])


class StateTokenTC(unittest.TestCase):
    """The seam that keeps a reply from landing on the wrong target."""

    def test_mismatch_drops_the_commands_and_ends_the_turn(self):
        runner = _Runner()
        session = _session(['[{"op": "add_circle"}]'], runner=runner)
        moved = [False]
        turn = agent.Turn(session, "draw", budget=4,
                          token=lambda s, scene: moved[0])
        request = turn.next_request()
        moved[0] = True  # the canvas changed while the backend was running
        turn.feed(request.send_to(session.backend))
        self.assertEqual(turn.stop_reason, agent.StopReason.STATE)
        self.assertEqual(runner.commands, [])
        last = session.transcript[-1]
        self.assertEqual(last.role, agent.HistoryFormatter.MARKER_ROLE)
        self.assertIn("changed mid-turn", last.text)

    def test_a_token_of_none_is_a_token_not_a_missing_request(self):
        # A GUI seam may have nothing to key on, and a turn whose token is
        # None must still feed rather than raise into a Qt slot.
        session = _session(["[]"])
        turn = agent.Turn(session, "draw", token=lambda s, scene: None)
        turn.feed(_ask(session, turn))
        self.assertEqual(turn.stop_reason, agent.StopReason.COMPLETED)

    def test_default_token_separates_two_blank_worlds(self):
        class _World:
            def describe_state(self, level="basic"):
                return '{"shapes": []}'

        session = agent.AgentSession(world=_World(), runner=_Runner())
        first = agent.default_token(session, session.scene_context())
        session.world = _World()
        second = agent.default_token(session, session.scene_context())
        # Identical scenes, different worlds: content alone cannot tell them
        # apart, so identity has to be in the token.
        self.assertNotEqual(first, second)


class TurnGuardTC(unittest.TestCase):
    def test_a_late_reply_after_stop_is_dropped_not_raised(self):
        # The worker's reply lands after the user hit Stop, in a Qt slot where
        # an exception has nowhere to go.  Nothing of it may be applied.
        runner = _Runner()
        session = _session(['[{"op": "add_circle"}]'], runner=runner)
        turn = agent.Turn(session, "draw")
        request = turn.next_request()
        turn.stop()
        self.assertIsNone(turn.feed(request.send_to(session.backend)))
        self.assertEqual(runner.commands, [])
        self.assertEqual([t.role for t in session.transcript], ["user"])

    def test_the_prompt_is_not_replayed_beside_itself(self):
        # From step 2 on the prompt is no longer the trailing user turn, so
        # the history would repeat what the request tail already carries.
        session = _session(['[{"op": "add_circle"}]', "[]"])
        agent.run_turn(session, "draw a circle")
        replayed = agent.format_history(session.backend.requests[1].history)
        self.assertNotIn("user:", replayed)
        self.assertIn("add_circle", replayed)


class RunTurnWrapperTC(unittest.TestCase):
    def test_it_loops_until_the_model_stops(self):
        runner = _Runner()
        session = _session(['[{"op": "add_circle"}]',
                            '[{"op": "add_circle"}]', "[]"], runner=runner)
        agent.run_turn(session, "draw two circles", budget=4)
        self.assertEqual(len(runner.commands), 2)
        self.assertEqual([t.role for t in session.transcript],
                         ["user", "agent", "agent", "agent"])

    def test_the_second_step_carries_the_first_step_results(self):
        session = _session(['[{"op": "add_circle"}]', "[]"])
        agent.run_turn(session, "draw a circle")
        second = session.backend.requests[1]
        replayed = agent.format_history(second.history)
        self.assertIn("add_circle", replayed)
        self.assertIn("shape_id", replayed)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
