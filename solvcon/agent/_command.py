# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Generic command framework for the Agent.

``Command`` subclasses are collected in a ``CommandSet`` that derives the
JSON Schema, validators, and tool definitions; a ``CommandProcessor`` runs
them against a target, and a ``CommandDispatcher`` routes ops across
families. Imports no Qt; concrete commands live in domain packages such as
:mod:`solvcon.agent.draw`.
"""

import copy
import dataclasses


class CommandError(ValueError):
    """A command failed schema validation or could not be applied."""


CRUD_CATEGORIES = ("create", "read", "update", "delete", "log")


class Command:
    """One command: op name, JSON Schema, and behavior."""

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
    # ``op`` is a required const; ``category`` is an annotation
    # validators ignore.
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


def _merge_default(default, value):
    """Recursively overlay ``value`` onto ``default``."""
    if not (isinstance(default, dict) and isinstance(value, dict)):
        return value
    out = {}
    for name, prop in default.items():
        out[name] = (_merge_default(prop, value[name]) if name in value
                     else copy.deepcopy(prop))
    for name, val in value.items():
        if name not in default:
            out[name] = val
    return out


class CommandSet:
    """A named family of commands with its derived schema and validators."""

    def __init__(self, title, description, categories=CRUD_CATEGORIES):
        self.title = title  # family name, used as the schema title
        self.description = description  # human summary of the family
        self.categories = tuple(categories)  # allowed op categories
        self.commands = {}
        self._built = None

    def register(self, command):
        """Add a command to the set; usable as a class decorator."""
        cls = command if isinstance(command, type) else type(command)
        if not command.op:
            raise ValueError(f"{cls.__name__} has no op name")
        if command.op in self.commands:
            raise ValueError(f"duplicate command op '{command.op}'")
        if command.category not in self.categories:
            raise ValueError(
                f"{command.op} has unknown category '{command.category}'")
        for name in command.optional:
            if name not in command.arguments:
                raise ValueError(
                    f"{command.op} optional '{name}' is not an argument")
        for name, prop in command.arguments.items():
            if (isinstance(prop, dict) and "default" in prop
                    and name not in command.optional):
                raise ValueError(
                    f"{command.op}.{name} is defaulted but required")
        self.commands[command.op] = cls() if command is cls else command
        self._built = None
        return command

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
        # TODO: cold-cache build races benignly (identical state); lock if
        # that changes.
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
        """Validate a list of commands."""
        if not isinstance(commands, list):
            raise CommandError("a script must be a list of commands")
        for command in commands:
            self.validate_command(command)
        return commands

    def apply_defaults(self, command):
        """Return a copy of ``command`` with optional args defaulted."""
        schema = self.command_schemas[command["op"]]
        out = dict(command)
        for name, prop in schema["properties"].items():
            if not isinstance(prop, dict) or "default" not in prop:
                continue
            default = prop["default"]
            if name not in out:
                out[name] = copy.deepcopy(default)
            else:
                out[name] = _merge_default(default, out[name])
        return out

    def tool_definitions(self):
        """Describe every command as an MCP-style tool definition."""
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

    def command_from_tool_call(self, name, arguments=None):
        """Rebuild a command dict from an MCP-style tool call."""
        return {**(arguments or {}), "op": name}

    def commands_by_category(self):
        """Group op names by category, in the set's category order."""
        grouped = {category: [] for category in self.categories}
        for op, cmd in self.commands.items():
            grouped[cmd.category].append(op)
        return grouped


# CommandSet attributes a fronting module re-exports so it doubles as its set.
COMMAND_API = (
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


def install_command_api(namespace, command_set):
    """Wire a module to double as ``command_set``'s command API.

    Pass the module's ``globals()`` as ``namespace``; installs
    ``__getattr__``/``__dir__`` and returns the name tuple for ``__all__``.
    """
    def __getattr__(name):
        if name in COMMAND_API:
            return getattr(command_set, name)
        raise AttributeError(
            f"module {namespace['__name__']!r} has no attribute {name!r}")

    def __dir__():
        return sorted(set(namespace) | set(COMMAND_API))

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
    return COMMAND_API


@dataclasses.dataclass
class CommandResult:
    """The outcome of applying one command."""
    op: str
    ok: bool
    value: object = None
    error: str = None


class CommandProcessor:
    """Run a ``CommandSet``'s commands against a ``target``, keeping a log."""

    def __init__(self, target, command_set, validate_results=False,
                 reraise=False):
        self.target = target  # object the commands mutate
        self.command_set = command_set  # family whose commands run here
        self.validate_results = validate_results  # check each result schema
        self.reraise = reraise  # let unexpected errors propagate, not capture
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
            self.command_set.validate_command(command)
        except CommandError as exc:
            return CommandResult(op, False, error=str(exc))
        return self._apply(op, command)

    def run_script(self, commands, stop_on_error=False):
        """Validate the whole script, then apply each command in order."""
        self.command_set.validate_script(commands)
        results = []
        for command in commands:
            result = self._apply(command["op"], command)
            results.append(result)
            if stop_on_error and not result.ok:
                break
        return results

    def _apply(self, op, command):
        """Apply the command to the target, returning a ``CommandResult``."""
        args = self.command_set.apply_defaults(command)
        try:
            value = self.command_set.commands[op].apply(
                self.target, args, self)
            if self.validate_results:
                self.command_set.validate_result(op, value)
        except CommandError as exc:
            return CommandResult(op, False, error=str(exc))
        except Exception as exc:
            if self.reraise:
                raise
            return CommandResult(
                op, False, error=f"{type(exc).__name__}: {exc}")
        return CommandResult(op, True, value=value)


class CommandDispatcher:
    """Route each command to the processor of the family owning its op."""

    def __init__(self, executors):
        self.executors = list(executors)  # per-family processors to route to
        self._routes = {}
        for executor in self.executors:
            for op in executor.command_set.commands:
                if op in self._routes:
                    raise ValueError(
                        f"duplicate command op '{op}' across families")
                self._routes[op] = executor

    def tool_definitions(self):
        """Concatenate every member family's tool definitions."""
        tools = []
        for executor in self.executors:
            tools.extend(executor.command_set.tool_definitions())
        return tools

    def _route(self, command):
        if not isinstance(command, dict):
            raise CommandError(
                f"command must be an object, got {type(command).__name__}")
        op = command.get("op")
        executor = self._routes.get(op)
        if executor is None:
            raise CommandError(
                f"unknown op '{op}'; valid ops: {sorted(self._routes)}")
        return executor

    def run(self, command):
        """Route and run one command, capturing routing failures."""
        try:
            executor = self._route(command)
        except CommandError as exc:
            op = command.get("op") if isinstance(command, dict) else None
            return CommandResult(op if isinstance(op, str) else "?",
                                 False, error=str(exc))
        return executor.run(command)

    def run_script(self, commands, stop_on_error=False):
        """Validate the whole script across families, then apply in order."""
        if not isinstance(commands, list):
            raise CommandError("a script must be a list of commands")
        routed = []
        for command in commands:
            executor = self._route(command)
            executor.command_set.validate_command(command)
            routed.append((executor, command))
        results = []
        for executor, command in routed:
            result = executor.run(command)
            results.append(result)
            if stop_on_error and not result.ok:
                break
        return results

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
