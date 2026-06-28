# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Apply validated Agent Draw commands to a ``World``.

The run loop, validation, logging, and result checking are the generic
``CommandExecutor`` in :mod:`solvcon.agent._command`; this subclass binds the
``DRAW`` command family and the injected renderer.
"""

from .. import _command as _cmd
from . import command


class Executor(_cmd.CommandExecutor):
    """Apply validated Agent Draw commands to a ``World``.

    ``renderer`` is a callable
    ``renderer(world, view, width, height, antialiasing) -> bytes`` supplied by
    the harness/MCP front-ends; with no renderer, ``render_png`` returns a
    failed result rather than touching a GUI.
    """

    def __init__(self, world, renderer=None, validate_results=False):
        super().__init__(world, command.DRAW,
                         validate_results=validate_results)
        self.renderer = renderer

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
