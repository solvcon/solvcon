# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent Draw front-end for the ``World`` API.

The command vocabulary lives in ``command.py`` (one ``Command`` subclass per
command); the schema documents, validators, and tool definitions are derived
from one private command set. This module delegates its command-family API to
that set, so the harness and MCP adapters ride on the same commands.
``Executor`` applies validated commands to a ``World``.
"""

from .._command import (  # noqa: F401
    CRUD_CATEGORIES,
    Command,
    CommandError,
    CommandResult,
)
from .command import _command_set
from .executor import Executor  # noqa: F401

_COMMAND_API = (
    "apply_defaults",
    "categories",
    "command_from_tool_call",
    "command_schemas",
    "commands",
    "commands_by_category",
    "description",
    "result_schemas",
    "schema",
    "title",
    "tool_definitions",
    "validate_command",
    "validate_result",
    "validate_script",
)


def __getattr__(name):
    if name in _COMMAND_API:
        return getattr(_command_set, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_COMMAND_API))


__all__ = (
    "CRUD_CATEGORIES",
    "Command",
    "CommandError",
    "CommandResult",
    "Executor",
    *_COMMAND_API,
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
