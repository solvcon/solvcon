# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Minimal fallback surface for the Agent.

Agent Draw (#965) owns the real ``World`` command schema and Executor.  Until
that package ships in-tree, this module is a thin, Qt-free stand-in so the
session and the tests have a working tool surface and runner: it carries only a
dummy no-op op.  :mod:`_core` prefers the agentdraw package whenever it is
importable, so this bridge is unused once #965 lands.
"""

import dataclasses


# TODO: remove this whole module once Agent Draw (#965) provides the real tool
# surface and Executor. The dummy op exists only so a backend and the tests
# have a trivial command to exercise the dispatch path headless.
_DUMMY_OP = "ping"
_DUMMY_TOOL = {
    "name": _DUMMY_OP,
    "description": "Test-only no-op that acknowledges without drawing.",
    "parameters": {},
}


@dataclasses.dataclass
class DrawResult:
    """Outcome of one command: the ``op``, whether it applied, and the reason
    when it did not."""

    op: str
    ok: bool = True
    error: str = None


def tool_definitions():
    """The fallback tool surface: only the dummy ``ping`` op."""
    return [dict(_DUMMY_TOOL)]


class Executor:
    """Apply fallback command dicts.

    Only the dummy ``ping`` op is known; any real drawing op is reported as a
    failed :class:`DrawResult` pointing at Agent Draw (#965) rather than
    raised, so one command never aborts a batch.  ``world`` and ``renderer``
    match the agentdraw Executor signature so :mod:`_core` builds either the
    same way; both are unused until the real ops land.
    """

    def __init__(self, world, renderer=None):
        self._world = world

    def run(self, command):
        op = command.get("op") if isinstance(command, dict) else None
        if op == _DUMMY_OP:
            return DrawResult(op, ok=True)
        return DrawResult(op or "?", ok=False,
                          error="op %r needs Agent Draw (#965)" % (op,))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
