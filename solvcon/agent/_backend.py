# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Pluggable AI backend abstraction for the Agent.

A backend turns a prompt (plus context and a command tool surface) into a
:class:`BackendResponse`: prose and a list of command dicts.
Backends register in a process-wide registry so a caller can list the usable
ones and let the user pick.  The module imports no Qt.  The offline
:class:`EchoBackend` keeps the registry non-empty so there is always a working
default.
"""

import abc
import dataclasses
import json


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
        user turn.  The tool dump is compact and key-sorted so the prefix stays
        byte-stable for a prompt cache and does not vary run to run."""
        tools = json.dumps(tool_surface or [], sort_keys=True,
                           separators=(",", ":"))
        return (
            "Available operations (tool definitions):\n%s\n\n"
            "Current scene:\n%s\n\nUser request:\n%s"
            % (tools, scene_context, prompt))


_REGISTRY = []


def register(backend):
    """Add a backend, replacing any with the same name (so a re-import does
    not duplicate the built-in entries)."""
    for index, existing in enumerate(_REGISTRY):
        if existing.name == backend.name:
            _REGISTRY[index] = backend
            return backend
    _REGISTRY.append(backend)
    return backend


def all_backends():
    """Every registered backend, in registration order (a copy)."""
    return list(_REGISTRY)


def available_backends():
    """Registered backends whose ``available()`` returns True."""
    return [b for b in _REGISTRY if b.available()]


def get_backend(name):
    """The registered backend with ``name``, or ``None`` if absent."""
    for backend in _REGISTRY:
        if backend.name == name:
            return backend
    return None


class EchoBackend(AgentBackend):
    """Offline backend that proposes no commands and echoes the prompt.

    It is always :meth:`available` and fully deterministic, so a caller, the
    tests, and a no-key demo always have a backend that exercises the whole
    pipeline without any external process.
    """

    name = "echo (offline)"

    def available(self):
        return True

    def send(self, prompt, scene_context, tool_surface):
        return BackendResponse(text="echo: %s" % prompt, commands=[])


register(EchoBackend())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
