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
import enum
import json

from . import _command


class TextFormatter:
    """Rendering helpers every request section shares.

    :meth:`one_line` is what keeps a foreign string on the line it was given:
    tool descriptions, model prose, and error text all land in a line-oriented
    payload, where an embedded newline would read as the next section.
    """

    @classmethod
    def literal(cls, value):
        # allow_nan=False: a non-finite float must raise rather than dump as
        # the NaN and Infinity tokens no JSON reader accepts.
        return json.dumps(value, separators=(",", ":"), allow_nan=False)

    @classmethod
    def one_line(cls, text):
        return " ".join(str(text).split())


class ToolSurfaceFormatter(TextFormatter):
    """The renderers that turn JSON Schema tool definitions into the compact
    per-op signatures the model reads."""

    SCALAR_TYPES = {"number": "num", "integer": "int", "string": "str",
                    "boolean": "bool", "null": "null"}

    BOUNDS = (("minimum", ">="), ("exclusiveMinimum", ">"),
              ("maximum", "<="), ("exclusiveMaximum", "<"))

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


class HistoryFormatter(TextFormatter):
    """Replay recorded turns into a capped history section."""

    #: The role of a turn that records a change in what the conversation is
    #: about rather than something said; rendered as a bare ``... text`` line.
    MARKER_ROLE = "marker"

    TEXT_CAP = 400
    PART_CAP = 240  # each half of a command line: the arguments, the outcome
    TURN_CAP = 2000
    REQUEST_CAP = 24000
    GAP_ALLOWANCE = 120  # room for dropped-run announcements
    PIN_REACH = 8  # turns back a failure may be pinned from

    @classmethod
    def clip(cls, text, cap):
        """``text`` cut to ``cap`` characters, naming how many it lost so a
        truncated payload never reads as a complete one."""
        if len(text) <= cap:
            return text
        return "%s...(+%d chars)" % (text[:cap], len(text) - cap)

    @classmethod
    def value_text(cls, value):
        """One JSON value, falling back to ``repr`` when it is not JSON
        (nothing promises a foreign runner returns one).  Only the fallback
        needs flattening: ``json.dumps`` escapes every control character, so a
        dump cannot carry a raw newline."""
        try:
            return cls.literal(value)
        except (TypeError, ValueError):
            # An unserializable object raises the first; a circular reference
            # or a non-finite float the second.
            return cls.one_line(repr(value))

    @classmethod
    def outcome_text(cls, result):
        if result is None:
            return "not run"
        if getattr(result, "ok", False):
            value = getattr(result, "value", None)
            if value is None:
                return "ok"
            return cls.clip("ok " + cls.value_text(value), cls.PART_CAP)
        error = getattr(result, "error", None) or "failed"
        return cls.clip("error: " + cls.one_line(error), cls.PART_CAP)

    @classmethod
    def command_line(cls, command, result):
        arguments = ({name: value for name, value in command.items()
                      if name != "op"}
                     if isinstance(command, dict) else command)
        return "  %s %s -> %s" % (
            _command.op_of(command),
            cls.clip(cls.value_text(arguments), cls.PART_CAP),
            cls.outcome_text(result))

    @classmethod
    def turn(cls, turn):
        """The role line always survives; excess commands are counted, not
        rendered."""
        role = cls.one_line(getattr(turn, "role", None) or "?")
        text = cls.one_line(getattr(turn, "text", None) or "")
        if role == cls.MARKER_ROLE:
            return "... %s" % cls.clip(text or "context changed", cls.TEXT_CAP)
        head = ("%s: %s" % (role, cls.clip(text, cls.TEXT_CAP)) if text
                else "%s:" % role)
        lines, used = [head], len(head) + 1
        commands = list(getattr(turn, "commands", None) or ())
        results = list(getattr(turn, "results", None) or ())
        for index, command in enumerate(commands):
            result = results[index] if index < len(results) else None
            line = cls.command_line(command, result)
            if used + len(line) + 1 > cls.TURN_CAP:
                lines.append("  ... %d more commands"
                             % (len(commands) - index))
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    @classmethod
    def failed(cls, turn):
        """Whether ``turn`` went wrong, by a command that did or by the
        ``failed`` flag a turn that never reached one carries: a backend that
        timed out or replied with malformed JSON leaves no result to read, and
        that is exactly the failure worth pinning."""
        if getattr(turn, "failed", False):
            return True
        return any(not getattr(result, "ok", False)
                   for result in getattr(turn, "results", None) or ())

    @classmethod
    def last_failure(cls, turns):
        """The index of the newest turn carrying a failure, or ``None``.

        Only the last :attr:`PIN_REACH` turns are searched.  A failure older
        than that is one the conversation has moved past, and pinning it would
        spend the budget contradicting the instruction to fix what failed
        rather than repeat it.
        """
        for index in reversed(range(max(0, len(turns) - cls.PIN_REACH),
                                    len(turns))):
            if cls.failed(turns[index]):
                return index
        return None

    @classmethod
    def fit(cls, turn, room):
        """``(block, room left)`` for ``turn``, or ``None`` when it does not
        fit in ``room``."""
        block = cls.turn(turn)
        if len(block) + 1 > room:
            return None
        return block, room - len(block) - 1

    @classmethod
    def gap(cls, dropped):
        return ["... %d turns dropped" % dropped] if dropped else []

    @classmethod
    def render(cls, history, used=0):
        """The conversation section: the recorded turns, oldest first, that
        fit in what ``used`` characters leave of :attr:`REQUEST_CAP`.

        Turns are taken newest first, so growth drops the oldest, and a recent
        turn carrying a failure (see :meth:`last_failure`) is taken before any
        of them: the model has to keep seeing the error it is being asked to
        fix even once the turn that raised it has aged out.  Every dropped run
        is announced, including one after the last kept turn (which a pinned
        failure can leave behind), so the model cannot read what is left as
        one unbroken conversation.  At most two runs can be dropped, one on
        each side of a pin, which is what :attr:`GAP_ALLOWANCE` holds room
        for.

        Only the history gives way here.  The tool surface and the scene are
        never cut, so a payload whose fixed parts already exceed
        :attr:`REQUEST_CAP` overshoots it with no history at all.
        """
        turns = list(history or ())
        room = cls.REQUEST_CAP - used - cls.GAP_ALLOWANCE
        blocks = {}
        pinned = cls.last_failure(turns)
        if pinned is not None:
            fitted = cls.fit(turns[pinned], room)
            if fitted is not None:
                blocks[pinned], room = fitted
        for index in reversed(range(len(turns))):
            if index in blocks:
                continue
            fitted = cls.fit(turns[index], room)
            if fitted is None:
                break
            blocks[index], room = fitted
        if not blocks:
            return ""
        lines, previous = [], -1
        for index in sorted(blocks):
            lines.extend(cls.gap(index - previous - 1))
            lines.append(blocks[index])
            previous = index
        lines.extend(cls.gap(len(turns) - previous - 1))
        return "\n".join(lines)


def format_history(history, used=0):
    """Module-level entry to :meth:`HistoryFormatter.render`."""
    return HistoryFormatter.render(history, used)


@dataclasses.dataclass(frozen=True)
class BackendSetting:
    """One user-tunable knob a backend advertises to a settings editor.  An
    empty ``choices`` means free text; otherwise the value must be one of the
    listed strings."""

    name: str
    label: str
    choices: tuple = ()
    default: str = ""
    tooltip: str = ""


@dataclasses.dataclass(frozen=True)
class TurnRequest:
    """The arguments of :meth:`AgentBackend.send` as one frozen object.

    A driver composes it on its own thread and hands it to a worker, so what
    the backend is asked cannot shift under it while the call runs.
    """

    prompt: str
    scene_context: str = ""
    tool_surface: list = dataclasses.field(default_factory=list)
    history: list = dataclasses.field(default_factory=list)

    def send_to(self, backend):
        return backend.send(self.prompt, self.scene_context,
                            self.tool_surface, self.history)


class TransportOutcome(enum.Enum):
    """How the exchange that carried a request ended.

    Only :attr:`OK` means the model answered; the rest say the reply never
    arrived, which is what tells a loop to abort instead of retry.
    """

    OK = "ok"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ParseStatus(enum.Enum):
    """What a model reply turned out to be once parsed.

    :attr:`EMPTY` and :attr:`PROSE` both yield no commands and must stay
    distinct: an explicit ``[]`` is the model saying the request is done,
    while prose is the model talking instead of acting.
    """

    COMMANDS = "commands"
    EMPTY = "empty"
    PROSE = "prose"
    MALFORMED = "malformed"


@dataclasses.dataclass
class BackendResponse:
    """One backend reply.

    ``status`` defaults to what ``commands`` and ``text`` imply, so a backend
    that only fills the older fields still reports a usable status.
    """

    text: str = ""
    commands: list = dataclasses.field(default_factory=list)
    error: str = None
    outcome: TransportOutcome = TransportOutcome.OK
    status: ParseStatus = None

    def __post_init__(self):
        if self.status is None:
            if self.commands:
                self.status = ParseStatus.COMMANDS
            elif self.text:
                self.status = ParseStatus.PROSE
            else:
                self.status = ParseStatus.EMPTY
        if (self.error and self.status is ParseStatus.EMPTY
                and self.outcome is TransportOutcome.OK):
            # An error carrying nothing else is a backend reporting failure
            # through the older field alone.  Left as ok it would be
            # indistinguishable from the model answering that it is done.
            self.outcome = TransportOutcome.TRANSPORT


class AgentBackend(abc.ABC):
    """Interface every AI backend implements: a stable :attr:`name`, an
    :meth:`available` check, and :meth:`send`.  The tiny surface lets a caller
    drive any backend from a background thread.  Every backend also shares one
    system instruction (:attr:`_INSTRUCTIONS`, sent as a real system prompt)
    and one user-payload layout (:meth:`_compose_user`), so a CLI and an HTTP
    backend never drift apart in what they ask the model.

    A backend may also advertise user-tunable knobs through
    :meth:`settings_spec`; the base class stores and validates their values, so
    a settings editor works the same for every backend.
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
        "Reading the scene and the conversation. The scene lists the shapes "
        "already on the canvas as \"#id type [x_min, y_min, x_max, y_max]\"; "
        "edit one through its id instead of drawing it again. Earlier turns "
        "are replayed with each command's outcome, either ok and its result "
        "or the error it failed with; build on what is already there, and "
        "fix what failed rather than repeating it. Long text is cut short "
        "and marked with the amount dropped, so ask for what you need again "
        "rather than trusting a cut line.\n"
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

    def settings_spec(self):
        """The knobs this backend exposes, as a sequence of
        :class:`BackendSetting`.  Empty by default: a backend opts in."""
        return ()

    @property
    def _settings(self):
        """The stored values, filled with the declared defaults on first use.

        Built on demand rather than in an ``__init__``, so a subclass that
        writes its own constructor cannot lose the settings by forgetting to
        chain up to this class.
        """
        values = self.__dict__.get("_setting_values")
        if values is None:
            values = {setting.name: setting.default
                      for setting in self.settings_spec()}
            self.__dict__["_setting_values"] = values
        return values

    def settings(self):
        """The current value of every knob, as a ``name -> value`` dict."""
        return dict(self._settings)

    def get_setting(self, name):
        return self._settings[name]

    def set_setting(self, name, value):
        """Store ``value`` for the knob ``name``, raising :class:`KeyError` for
        an unknown knob and :class:`ValueError` for a value that is not a
        string or falls outside the knob's choices.

        An emptied free-text knob stores the declared default instead.  A
        backend reading a blank address or model would answer
        :meth:`available` with False and drop out of the selector that is the
        only way back to the settings editor.  Storing the default rather
        than falling back on each read also keeps one value: what the editor
        shows, what the configuration file records, and what the backend uses
        cannot disagree.
        """
        for setting in self.settings_spec():
            if setting.name != name:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    "%s: %s takes a string, not %r" % (self.name, name, value))
            if setting.choices and value not in setting.choices:
                raise ValueError(
                    "%s: %r is not a valid %s" % (self.name, value, name))
            self._settings[name] = value or setting.default
            return
        raise KeyError("%s has no setting %r" % (self.name, name))

    @abc.abstractmethod
    def available(self):
        """Whether this backend can run now (CLI on PATH, key set, ...)."""

    @abc.abstractmethod
    def send(self, prompt, scene_context, tool_surface, history=()):
        """Run the backend and return a :class:`BackendResponse`.

        :param prompt: the user's natural-language request.
        :param scene_context: a short text summary of the current world.
        :param tool_surface: the command tool definitions the model may call.
        :param history: the recorded turns to replay, oldest first, each
            exposing ``role``, ``text``, ``commands``, and ``results``.
        """

    _SURFACE = "Available operations:\n%s\n\n"
    _HEADER = "Conversation so far:\n"
    _GAP = "\n\n"
    _TAIL = "Current scene:\n%s\n\nUser request:\n%s"

    @classmethod
    def _sections(cls, prompt, scene_context, tool_surface, history):
        """The three pieces of the user payload: the rendered tool surface,
        the conversation that fits beside it, and the scene-and-request
        tail."""
        surface = cls._SURFACE % format_tool_surface(tool_surface)
        tail = cls._TAIL % (scene_context, prompt)
        story = format_history(
            history,
            used=len(surface) + len(cls._HEADER) + len(cls._GAP) + len(tail))
        return surface, story, tail

    @classmethod
    def history_section(cls, prompt, scene_context, tool_surface, history=()):
        """The conversation section this request will carry, for a caller that
        wants to show what was replayed.  It comes from the same
        :meth:`_sections` the payload is built from, so what is shown and what
        is sent cannot drift apart."""
        return cls._sections(prompt, scene_context, tool_surface, history)[1]

    @classmethod
    def _compose_user(cls, prompt, scene_context, tool_surface, history=()):
        """The user-role payload: the tool surface, the conversation so far,
        the scene, and the request.  The shared instruction rides separately
        as the system prompt (:attr:`_INSTRUCTIONS`), so it stays a stable
        prefix a backend can hand the model as a real system message rather
        than folding it into the user turn.

        The scene and the request come last because they are what the model
        answers.
        """
        surface, story, tail = cls._sections(
            prompt, scene_context, tool_surface, history)
        if not story:
            return surface + tail
        return surface + cls._HEADER + story + cls._GAP + tail


class BackendRegistry:
    """The process-wide list of backends a caller can offer the user.

    The entries are class state, so every importer shares one registry and
    registration order is the order a selector shows.
    """

    _BACKENDS = []

    #: The configuration key holding every backend's settings, as a
    #: ``backend name -> {knob: value}`` mapping.
    CONFIG_KEY = "agent_backend_settings"

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

    @classmethod
    def load_settings(cls, config):
        """Apply the settings stored under :attr:`CONFIG_KEY` to the registered
        backends.

        An entry naming a backend, a knob, or a value the running code does not
        know is dropped: the configuration file outlives any one version, and a
        stale entry must not keep the console from starting.
        """
        stored = config.get(cls.CONFIG_KEY)
        if not isinstance(stored, dict):
            return
        for backend in cls._BACKENDS:
            values = stored.get(backend.name)
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                try:
                    backend.set_setting(name, value)
                except (KeyError, ValueError):
                    continue

    @classmethod
    def save_settings(cls, config):
        """Record every backend's settings under :attr:`CONFIG_KEY`.  Writing
        the file is the caller's call, so a caller can batch several edits into
        one :meth:`~solvcon.config.Config.save`."""
        # Merge rather than replace: an entry belongs to a backend this process
        # never registered (renamed, or from another build), and rewriting the
        # key from the live list alone would delete settings the user still
        # wants for it.
        stored = config.get(cls.CONFIG_KEY)
        stored = dict(stored) if isinstance(stored, dict) else {}
        stored.update({backend.name: backend.settings()
                       for backend in cls._BACKENDS if backend.settings()})
        config.set(cls.CONFIG_KEY, stored)


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

    def send(self, prompt, scene_context, tool_surface, history=()):
        return BackendResponse(text="echo: %s" % prompt, commands=[])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
