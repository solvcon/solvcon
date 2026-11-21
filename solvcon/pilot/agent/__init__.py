# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The controlling-agent GUI on top of the headless :mod:`solvcon.agent`
session: the dock panel that drives the 2D world from natural language, the
turn runner that runs a prompt without blocking the window, and the backend
settings.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_agent_gui': ['AgentPanel'],
    '_agent_settings': ['edit_backend_settings'],
    '_agent_turn': ['AgentTurnRunner', 'commands_as_python', 'format_command',
                    'undo_steps', 'undo_turn_steps'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
