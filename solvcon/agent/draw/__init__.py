# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Agent Draw front-end for the ``World`` API.

The command vocabulary lives in ``command.py`` (one ``Command`` subclass per
command, collected in ``DRAW``); ``Executor`` applies validated commands to a
``World``. The harness and MCP adapters ride on the same commands.
"""

from .._command import (  # noqa: F401
    CRUD_CATEGORIES,
    Command,
    CommandError,
    CommandResult,
)
from .command import (  # noqa: F401
    COMMANDS,
    COMMAND_SCHEMAS,
    RESULT_SCHEMAS,
    SCHEMA,
    apply_defaults,
    commands_by_category,
    tool_definitions,
    validate_command,
    validate_result,
    validate_script,
)
from .executor import Executor  # noqa: F401

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
