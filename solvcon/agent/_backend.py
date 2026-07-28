# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Pluggable AI backend abstraction for the Agent.

A backend turns a prompt (plus context and a command tool surface) into a
:class:`BackendResponse`: prose and a list of command dicts.
Backends register in a process-wide registry so a caller can list the usable
ones and let the user pick.  The first registration is what a selector starts
on.  The module imports no Qt.  The offline :class:`EchoBackend` stays out of
the registry: it is a test and demo double, not a backend to offer a user.
"""

import abc
import dataclasses
import json


class ToolSurfaceFormatter:
    """The renderers that turn JSON Schema tool definitions into the compact
    per-op signatures the model reads."""

    SCALAR_TYPES = {"number": "num", "integer": "int", "string": "str",
                    "boolean": "bool", "null": "null"}

    BOUNDS = (("minimum", ">="), ("exclusiveMinimum", ">"),
              ("maximum", "<="), ("exclusiveMaximum", "<"))

    @classmethod
    def literal(cls, value):
        return json.dumps(value, separators=(",", ":"))

    @classmethod
    def one_line(cls, text):
        return " ".join(str(text).split())

    @classmethod
    def array_type(cls, schema):
        item = cls.type_name(schema.get("items", {}))
        low, high = schema.get("minItems"), schema.get("maxItems")
        if low is None and high is None:
            return "[%s]" % item
        if low == high:
            return "[%s]x%d" % (item, low)
        if high is None:
            return "[%s]x%d+" % (item, low)
        if low is None:
            return "[%s]x..%d" % (item, high)
        return "[%s]x%d..%d" % (item, low, high)

    @classmethod
    def fields(cls, schema):
        properties = schema.get("properties") or {}
        required = set(schema.get("required", ()))
        fields = []
        for name, prop in properties.items():
            text = "%s%s: %s" % (name, "" if name in required else "?",
                                 cls.type_name(prop))
            if isinstance(prop, dict) and "default" in prop:
                text += " = %s" % cls.literal(prop["default"])
            fields.append(text)
        return ", ".join(fields)

    @classmethod
    def object_type(cls, schema):
        """``{name: T}`` over the declared properties."""
        return "{%s}" % cls.fields(schema)

    @classmethod
    def type_name(cls, schema):
        """Render one JSON Schema fragment as a compact type expression."""
        if not isinstance(schema, dict):
            return "any"
        if "const" in schema:
            return cls.literal(schema["const"])
        if "enum" in schema:
            return "|".join(
                cls.literal(value) for value in schema["enum"]) or "any"
        kind = schema.get("type")
        if isinstance(kind, list):
            # A union of types is a list; each member carries the same bounds.
            return "|".join(cls.type_name({**schema, "type": one})
                            for one in kind) or "any"
        if kind == "array":
            return cls.array_type(schema)
        if kind == "object":
            return cls.object_type(schema)
        if kind in ("number", "integer"):
            return cls.SCALAR_TYPES[kind] + "".join(
                mark + cls.literal(schema[key])
                for key, mark in cls.BOUNDS if key in schema)
        if kind == "string" and schema.get("contentEncoding"):
            return "str(%s)" % schema["contentEncoding"]
        return cls.SCALAR_TYPES.get(kind, "any")

    @classmethod
    def signature(cls, tool):
        """One op's call signature: name, arguments, and the result object it
        returns."""
        line = "%s(%s)" % (tool.get("name", "?"),
                           cls.fields(tool.get("inputSchema") or {}))
        returns = tool.get("outputSchema") or {}
        if returns.get("properties"):
            line += " -> %s" % cls.type_name(returns)
        return line

    @classmethod
    def prose(cls, tool):
        """The op's description followed by one line per described
        argument."""
        lines = []
        if tool.get("description"):
            lines.append(tool["description"])
        properties = (tool.get("inputSchema") or {}).get("properties") or {}
        for name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            description = prop.get("description")
            if description:
                lines.append("%s: %s" % (name, description))
        return lines

    @classmethod
    def render(cls, tool_surface):
        """Render tool definitions as compact per-op signatures."""
        lines = []
        current = None
        for tool in tool_surface or []:
            category = cls.one_line(tool.get("category") or "other")
            if category != current:
                lines.append("[%s]" % category)
                current = category
            lines.append(cls.one_line(cls.signature(tool)))
            lines.extend("  " + cls.one_line(line) for line in cls.prose(tool))
        return "\n".join(lines)


def format_tool_surface(tool_surface):
    """Module-level entry to :meth:`ToolSurfaceFormatter.render`."""
    return ToolSurfaceFormatter.render(tool_surface)


@dataclasses.dataclass
class BackendResponse:
    """One backend reply: ``text`` prose, the proposed ``commands`` the
    session applies, and an ``error`` reason or ``None``."""

    text: str = ""
    commands: list = dataclasses.field(default_factory=list)
    error: str = None


class AgentBackend(abc.ABC):
    """Interface every AI backend implements: a stable :attr:`name`, an
    :meth:`available` check, and :meth:`send`.  The tiny surface lets a caller
    drive any backend from a background thread.  Every backend also shares one
    system instruction (:attr:`_INSTRUCTIONS`, sent as a real system prompt)
    and one user-payload layout (:meth:`_compose_user`), so a CLI and an HTTP
    backend never drift apart in what they ask the model.
    """

    # TODO: solvcon is an application platform for geometry-based computation:
    # editing graphics and geometry, visualizing, meshing, and solving
    # conservation laws.  These instructions frame the agent around the 2D
    # drawing canvas alone; reframe that as one capability among the platform's
    # rather than the agent's whole scope.  Related to #966.
    _INSTRUCTIONS = (
        "You drive a 2D drawing canvas along with its windows and view. "
        "Turn the user's request into a JSON array of commands chosen from "
        "the operations below: draw on the canvas, open or arrange canvas "
        "windows, and pan or zoom the view. Reply with only that array.\n"
        "\n"
        "Reading the operation list. Operations are listed under a "
        "category header and given as a signature: op(arg: type, "
        "optional?: type = default) -> {result: type}. An argument marked "
        "? may be omitted and takes the default shown. Types are num, int, "
        "str, bool, null, and any; str(base64) is base64-encoded text; one "
        "or more comparisons follow a number as in num>0 or int>=1<=9; "
        "[T] is an array, [T]xN an array of "
        "exactly N items, [T]xN+ at least N, [T]x..M at most M, [T]xN..M "
        "between N and M; "
        "{a: T} is an object with those fields; \"a\"|\"b\" lists the only "
        "allowed values. The indented lines under a signature describe the "
        "operation and then its arguments. Pass no key an operation does "
        "not list: an unlisted key is rejected, not ignored.\n"
        "\n"
        "Coordinate frame (canvas drawing). The canvas uses world "
        "coordinates with the origin (0, 0) at the center and +Y pointing "
        "up, so a larger y is higher on screen. The origin is always in "
        "view; keep the whole subject centered on it and within about x in "
        "[-180, 180] and y in [-130, 130] so nothing is clipped, letting it "
        "span a couple hundred units to fill the canvas. Do not draw into a "
        "small first-quadrant box such as x in [0, 100]: this is centered "
        "world space, not screen or SVG pixels where (0, 0) is a corner and "
        "y grows downward.\n"
        "\n"
        "Plan first. When drawing, break the subject into parts and choose "
        "each part's size and position in world units before emitting "
        "commands, so the parts line up and stay in frame. You may record "
        "the plan as a leading \"log\" command.\n"
        "\n"
        "Compose cleanly. Keep repeated elements (wheels, petals, letters) "
        "consistent in size and spacing, and add shapes back to front so "
        "nearer parts are drawn last.\n"
        "\n"
        "Output contract. Each command is an object with an \"op\" key "
        "naming the operation and the operation's arguments as sibling "
        "keys. Reply with only the JSON array, no prose and no code fences. "
        "Use an empty array when the request needs no action.\n"
        "\n"
        "Example, for \"draw a simple house\":\n"
        "[\n"
        "  {\"op\": \"log\", \"message\": \"body, roof, door\"},\n"
        "  {\"op\": \"add_rectangle\", \"x_min\": -100, \"y_min\": -110, "
        "\"x_max\": 100, \"y_max\": 40},\n"
        "  {\"op\": \"add_triangle\", \"x0\": -125, \"y0\": 40, "
        "\"x1\": 125, \"y1\": 40, \"x2\": 0, \"y2\": 125},\n"
        "  {\"op\": \"add_rectangle\", \"x_min\": -25, \"y_min\": -110, "
        "\"x_max\": 25, \"y_max\": -20}\n"
        "]"
    )

    @property
    @abc.abstractmethod
    def name(self):
        """Short, stable identifier shown in the backend selector."""

    @abc.abstractmethod
    def available(self):
        """Whether this backend can run now (CLI on PATH, key set, ...)."""

    @abc.abstractmethod
    def send(self, prompt, scene_context, tool_surface):
        """Run the backend and return a :class:`BackendResponse`.

        :param prompt: the user's natural-language request.
        :param scene_context: a short text summary of the current world.
        :param tool_surface: the command tool definitions the model may call.
        """

    @classmethod
    def _compose_user(cls, prompt, scene_context, tool_surface):
        """The user-role payload: the tool surface, scene, and request.  The
        shared instruction rides separately as the system prompt
        (:attr:`_INSTRUCTIONS`), so it stays a stable prefix a backend can hand
        the model as a real system message rather than folding it into the
        user turn."""
        return (
            "Available operations:\n%s\n\n"
            "Current scene:\n%s\n\nUser request:\n%s"
            % (format_tool_surface(tool_surface), scene_context, prompt))


class BackendRegistry:
    """The process-wide list of backends a caller can offer the user.

    The entries are class state, so every importer shares one registry and
    registration order is the order a selector shows.
    """

    _BACKENDS = []

    @classmethod
    def register(cls, backend):
        """Add a backend, replacing any with the same name (so a re-import does
        not duplicate the built-in entries)."""
        for index, existing in enumerate(cls._BACKENDS):
            if existing.name == backend.name:
                cls._BACKENDS[index] = backend
                return backend
        cls._BACKENDS.append(backend)
        return backend

    @classmethod
    def all(cls):
        """Every registered backend, in registration order (a copy)."""
        return list(cls._BACKENDS)

    @classmethod
    def available(cls):
        """Registered backends whose ``available()`` returns True, in
        registration order, so the first entry is what a selector defaults
        to."""
        return [b for b in cls._BACKENDS if b.available()]

    @classmethod
    def get(cls, name):
        """The registered backend with ``name``, or ``None`` if absent."""
        for backend in cls._BACKENDS:
            if backend.name == name:
                return backend
        return None


class EchoBackend(AgentBackend):
    """Offline backend that proposes no commands and echoes the prompt.

    It is always :meth:`available` and fully deterministic, so the tests and a
    no-key demo can exercise the whole pipeline without any external process.
    It does not register itself: a user picks a real backend, and a caller that
    wants this one instantiates or registers it explicitly.
    """

    name = "echo (offline)"

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface):
        return BackendResponse(text="echo: %s" % prompt, commands=[])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
