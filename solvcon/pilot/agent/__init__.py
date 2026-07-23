# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The controlling-agent GUI: the dock panel that drives the 2D world from
natural language on top of the headless :mod:`solvcon.agent` session.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _agent_gui

    AgentPanel = _agent_gui.AgentPanel
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    AgentPanel = None

__all__ = [
    'AgentPanel',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
