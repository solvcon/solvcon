# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent Window front-end for the pilot canvas windowing API.
"""

from .. import _command as _cmd
from .._command import (  # noqa: F401
    CRUD_CATEGORIES,
    Command,
    CommandError,
    CommandResult,
)
from .command import _command_set
from .executor import Executor  # noqa: F401

__all__ = (
    "CRUD_CATEGORIES",
    "Command",
    "CommandError",
    "CommandResult",
    "Executor",
    *_cmd.install_command_api(globals(), _command_set),
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
