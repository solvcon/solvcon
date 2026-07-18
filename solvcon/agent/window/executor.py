# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Apply validated Agent Window commands to a window manager."""

from .. import _command as _cmd
from . import command


class Executor(_cmd.CommandProcessor):
    """Bind the window command set to a window-manager target."""

    def __init__(self, manager, validate_results=False, reraise=False):
        super().__init__(manager, command._command_set,
                         validate_results=validate_results, reraise=reraise)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
