# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The controlling-agent GUI: the dock panel that drives the 2D world from
natural language on top of the headless :mod:`solvcon.agent` session.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_agent_gui': ['AgentPanel'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
