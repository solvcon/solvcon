# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Apply validated Agent Draw commands to a ``World``.

The run loop, validation, logging, and result checking are the generic
``CommandProcessor`` in :mod:`solvcon.agent._command`; this subclass binds the
Agent Draw command set and the injected renderer.
"""

from .. import _command as _cmd
from . import command


class Executor(_cmd.CommandProcessor):
    """Apply validated Agent Draw commands to a ``World``.

    ``renderer`` is a callable
    ``renderer(world, view, width, height, antialiasing) -> bytes`` supplied by
    the harness/MCP front-ends; with no renderer, ``render_png`` returns a
    failed result rather than touching a GUI.
    """

    def __init__(self, world, renderer=None, validate_results=False,
                 reraise=False):
        super().__init__(world, command._command_set,
                         validate_results=validate_results, reraise=reraise)
        self.renderer = renderer

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
