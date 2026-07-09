# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent: drive the 2D ``World`` with an AI backend, with or without a GUI.

This package lives outside :mod:`solvcon.pilot` so it can also drive pure
computation that needs no graphics.  The headless core (:mod:`_core`) and the
backend abstraction (:mod:`_backend`) load without Qt, so they run in CI and a
headless build.
"""

from . import _core  # noqa: F401
from . import _backend  # noqa: F401
from . import _backends_impl  # noqa: F401

# _core.py
list_of_core = [
    'AgentSession',
    'TranscriptTurn',
]

# _backend.py
list_of_backend = [
    'AgentBackend',
    'BackendResponse',
    'EchoBackend',
    'register',
    'all_backends',
    'available_backends',
    'get_backend',
]

# _backends_impl.py
list_of_backends_impl = [
    'SubprocessBackend',
    'ClaudeCliBackend',
    'OpenAIHttpBackend',
    'parse_tool_calls',
]

# TODO: when the Qt dock module exists in solvcon.pilot, point this at its
# Agent class, guarded by _pilot_core.enable like the airfoil sub-package.
Agent = None

__all__ = (  # noqa: F822
    list_of_core + list_of_backend + list_of_backends_impl + ['Agent']
)


def _load(module, symbol_list):
    for name in symbol_list:
        globals()[name] = getattr(module, name)


_load(_core, list_of_core)
_load(_backend, list_of_backend)
_load(_backends_impl, list_of_backends_impl)

del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
