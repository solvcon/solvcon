# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Generic command framework for the Agent.

A command family (drawing, mesh, pilot control) is a set of ``Command``
subclasses collected in a ``CommandSet``, which derives the JSON Schema, the
validators, and the tool definitions from them; a ``CommandExecutor`` runs
validated commands against a target. This mirrors the backend registry in
:mod:`_backend` and imports no Qt. Domain packages such as
:mod:`solvcon.agent.draw` supply the concrete commands.
"""

import copy
import dataclasses


class CommandError(ValueError):
    """A command failed schema validation or could not be applied."""


CRUD_CATEGORIES = ("create", "read", "update", "delete", "log")


# Shared JSON Schema fragments, used by reference; treat as immutable.
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}
INTEGER = {"type": "integer"}
POSITIVE_INT = {"type": "integer", "exclusiveMinimum": 0}
BOOLEAN = {"type": "boolean"}
STRING = {"type": "string"}


# Property builders that attach a description; models call tools more reliably
# when each argument is described.
def _num(description):
    return {**NUMBER, "description": description}


def _pos(description):
    return {**POSITIVE, "description": description}


def _int(description):
    return {**INTEGER, "description": description}


class Command:
    """One command: its op name, JSON Schema, and behavior.

    A subclass sets ``op``, ``category`` (one of ``CRUD_CATEGORIES``),
    ``summary``, ``arguments`` (name -> JSON Schema property), ``optional``
    (argument names that may be omitted, each free to carry a ``default``), and
    ``returns`` (name -> property), then implements ``apply``.
    """

    op = ""
    category = ""
    summary = ""
    arguments = {}
    optional = ()
    returns = {}

    def apply(self, target, args, ctx):
        """Apply the command to ``target`` and return its result mapping."""
        raise NotImplementedError


def _command_schema(cmd):
    # ``op`` is pinned to a const and always required; ``category`` rides along
    # as an annotation keyword validators ignore.
    required = [name for name in cmd.arguments if name not in cmd.optional]
    return {
        "title": cmd.op,
        "description": cmd.summary,
        "category": cmd.category,
        "type": "object",
        "properties": {"op": {"const": cmd.op}, **cmd.arguments},
        "required": ["op"] + required,
        "additionalProperties": False,
    }


def _result_schema(cmd):
    return {
        "title": f"{cmd.op}_result",
        "type": "object",
        "properties": dict(cmd.returns),
        "required": list(cmd.returns),
        "additionalProperties": False,
    }


class CommandSet:
    """A named family of commands with its derived schema and validators.

    Register ``Command`` subclasses with :meth:`register` (as a decorator); the
    schema documents, compiled validators, and tool definitions are derived
    lazily and cached, and every :meth:`register` invalidates the cache.
    """

    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.commands = {}
        self._built = None

    def register(self, cls):
        """Register a ``Command`` subclass, checking its declaration."""
        if not cls.op:
            raise ValueError(f"{cls.__name__} has no op name")
        if cls.op in self.commands:
            raise ValueError(f"duplicate command op '{cls.op}'")
        if cls.category not in CRUD_CATEGORIES:
            raise ValueError(
                f"{cls.op} has unknown category '{cls.category}'")
        for name in cls.optional:
            if name not in cls.arguments:
                raise ValueError(
                    f"{cls.op} optional '{name}' is not an argument")
        for name, prop in cls.arguments.items():
            if (isinstance(prop, dict) and "default" in prop
                    and name not in cls.optional):
                raise ValueError(f"{cls.op}.{name} is defaulted but required")
        self.commands[cls.op] = cls()
        self._built = None
        return cls

    def _build(self):
        import jsonschema
        command_schemas = {op: _command_schema(cmd)
                           for op, cmd in self.commands.items()}
        result_schemas = {op: _result_schema(cmd)
                          for op, cmd in self.commands.items()}
        return {
            "command_schemas": command_schemas,
            "result_schemas": result_schemas,
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": self.title,
                "description": self.description,
                "oneOf": list(command_schemas.values()),
            },
            "validators": {op: jsonschema.Draft202012Validator(schema)
                           for op, schema in command_schemas.items()},
            "result_validators": {op: jsonschema.Draft202012Validator(schema)
                                  for op, schema in result_schemas.items()},
        }

    @property
    def _cache(self):
        if self._built is None:
            self._built = self._build()
        return self._built

    @property
    def command_schemas(self):
        return self._cache["command_schemas"]

    @property
    def result_schemas(self):
        return self._cache["result_schemas"]

    @property
    def schema(self):
        return self._cache["schema"]

    def _validate(self, kind, op, value, prefix):
        import jsonschema
        validator = self._cache[kind].get(op)
        if validator is None:
            raise CommandError(
                f"unknown op '{op}'; valid ops: {sorted(self.commands)}")
        try:
            validator.validate(value)
        except jsonschema.ValidationError as exc:
            raise CommandError(f"{prefix}{exc.message}") from exc
        return value

    def validate_command(self, command):
        """Validate ``command`` against its op's JSON Schema."""
        if not isinstance(command, dict):
            raise CommandError(
                f"command must be an object, got {type(command).__name__}")
        op = command.get("op")
        if not isinstance(op, str):
            raise CommandError("command is missing a string 'op' field")
        self._validate("validators", op, command, f"{op}: ")
        return command

    def validate_result(self, op, value):
        """Validate a command's result against its declared output schema."""
        return self._validate(
            "result_validators", op, value, f"{op} result: ")

    def validate_script(self, commands):
        """Validate a list of commands, returning it unchanged on success."""
        if not isinstance(commands, list):
            raise CommandError("a script must be a list of commands")
        for command in commands:
            self.validate_command(command)
        return commands

    def apply_defaults(self, command):
        """Return a copy of ``command`` with omitted optional args filled in.

        Validators do not apply schema defaults, so the executor calls this
        after validating. A dict-valued default is merged field by field, so a
        partial object reaches the command complete.
        """
        schema = self.command_schemas[command["op"]]
        out = dict(command)
        for name, prop in schema["properties"].items():
            if not isinstance(prop, dict) or "default" not in prop:
                continue
            default = prop["default"]
            if name not in out:
                out[name] = copy.deepcopy(default)
            elif isinstance(default, dict) and isinstance(out[name], dict):
                out[name] = {**default, **out[name]}
        return out

    def tool_definitions(self):
        """Describe every command as an MCP-style name and I/O schema.

        The ``op`` field is dropped because the tool name already carries it.
        """
        tools = []
        for op, schema in self.command_schemas.items():
            properties = {name: prop
                          for name, prop in schema["properties"].items()
                          if name != "op"}
            required = [name for name in schema["required"] if name != "op"]
            tools.append({
                "name": op,
                "category": schema["category"],
                "description": schema["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "outputSchema": self.result_schemas[op],
            })
        return tools

    def commands_by_category(self):
        """Group op names by their CRUD category, in canonical order."""
        grouped = {category: [] for category in CRUD_CATEGORIES}
        for op, cmd in self.commands.items():
            grouped[cmd.category].append(op)
        return grouped


@dataclasses.dataclass
class CommandResult:
    """The outcome of applying one command."""
    op: str
    ok: bool
    value: object = None
    error: str = None


class CommandExecutor:
    """Run a ``CommandSet``'s commands against a ``target``, recording a log.

    Errors are captured into a failed ``CommandResult`` rather than raised, so
    a recording harness can log every step. With ``validate_results=True`` each
    success value is checked against its op's result schema; it is off by
    default. A family needing per-target context subclasses this and adds it
    as an attribute the commands read off ``ctx``.
    """

    def __init__(self, target, commands, validate_results=False):
        self.target = target
        self.commands = commands
        self.validate_results = validate_results
        self._log = []

    def append_log(self, message):
        self._log.append(message)

    @property
    def log(self):
        return list(self._log)

    def run(self, command):
        """Validate and apply one command, returning its ``CommandResult``."""
        op = command.get("op", "?") if isinstance(command, dict) else "?"
        try:
            self.commands.validate_command(command)
        except CommandError as exc:
            return CommandResult(op, False, error=str(exc))
        return self._apply(op, command)

    def run_script(self, commands):
        """Validate the whole script up front, then apply each in order."""
        self.commands.validate_script(commands)
        return [self._apply(c["op"], c) for c in commands]

    def _apply(self, op, command):
        args = self.commands.apply_defaults(command)
        try:
            value = self.commands.commands[op].apply(self.target, args, self)
            if self.validate_results:
                self.commands.validate_result(op, value)
        except CommandError as exc:
            return CommandResult(op, False, error=str(exc))
        except Exception as exc:
            return CommandResult(
                op, False, error=f"{type(exc).__name__}: {exc}")
        return CommandResult(op, True, value=value)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
