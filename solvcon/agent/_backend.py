# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Pluggable AI backend abstraction for the Agent.

A backend turns a prompt (plus scene context and the Agent Draw tool surface)
into a :class:`BackendResponse`: prose and a list of Agent Draw command dicts.
Backends register in a process-wide registry so a caller can list the usable
ones and let the user pick.  The module imports no Qt.  The offline
:class:`EchoBackend` keeps the registry non-empty so there is always a working
default.
"""

import abc
import dataclasses


@dataclasses.dataclass
class BackendResponse:
    """One backend reply: ``text`` prose, the proposed ``commands`` (Agent
    Draw dicts the session applies; empty means no drawing), and an ``error``
    reason or ``None``."""

    text: str = ""
    commands: list = dataclasses.field(default_factory=list)
    error: str = None


class AgentBackend(abc.ABC):
    """Interface every AI backend implements: a stable :attr:`name`, an
    :meth:`available` check, and :meth:`send`.  The tiny surface lets a caller
    drive any backend from a background thread."""

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
        :param tool_surface: the Agent Draw tool definitions the model may
            call.
        """


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
    """Offline backend that proposes no drawing and echoes the prompt.

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
