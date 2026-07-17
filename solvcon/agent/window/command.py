# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Agent Window command family: the canvas windowing surface for an agent.

The commands drive a window manager, a duck-typed target the GUI
``WindowController`` will bind to the pilot's ``RManager`` and ``QMdiArea``
(tests bind an in-memory fake). The manager surface:

    new_canvas() -> int
        Open a new 2D canvas window and return its stable id.
    list_windows() -> list of {"id": int, "title": str, "active": bool}
        Every open canvas window, in tab order.
    activate_window(window_id) -> None
        Make the window with the given id active.
    close_window(window_id) -> None
        Close the window with the given id.
    save_image(window_id, path) -> None
        Save the window's canvas image to ``path``.
"""

from .. import _command as _cmd


# Shared JSON Schema fragments, used by reference; treat as immutable.
INTEGER = {"type": "integer"}
BOOLEAN = {"type": "boolean"}
STRING = {"type": "string"}


def _int(description):
    return {**INTEGER, "description": description}


WINDOW = {"type": "object",
          "description": "One open canvas window: id, title, active flag.",
          "properties": {
              "id": _int("Stable id of the canvas window."),
              "title": {**STRING, "description": "Window title text."},
              "active": {**BOOLEAN,
                         "description": "True for the active window."}},
          "required": ["id", "title", "active"],
          "additionalProperties": False}


def _require_window(manager, window_id):
    # Fail by-id commands cleanly instead of leaking a KeyError from the
    # manager or the Qt layer.
    if not any(w["id"] == window_id for w in manager.list_windows()):
        raise _cmd.CommandError(f"no open window with id {window_id}")


_command_set = _cmd.CommandSet(
    "Agent Window command", "Any single command in the Agent Window schema.")


@_command_set.register
class NewCanvas(_cmd.Command):
    op = "new_canvas"
    category = "create"
    summary = "Open a new 2D canvas window and make it the active one."
    returns = {"window_id": _int("Id of the newly opened window.")}

    def apply(self, manager, args, ctx):
        return {"window_id": manager.new_canvas()}


@_command_set.register
class ListWindows(_cmd.Command):
    op = "list_windows"
    category = "read"
    summary = "List every open canvas window with id, title, and active flag."
    returns = {"windows": {"type": "array", "items": WINDOW,
                           "description": "Open canvas windows, tab order."}}

    def apply(self, manager, args, ctx):
        return {"windows": list(manager.list_windows())}


@_command_set.register
class ActivateWindow(_cmd.Command):
    op = "activate_window"
    category = "update"
    summary = "Make the canvas window with the given id the active one."
    arguments = {"window_id": _int("Id of the window to activate.")}

    def apply(self, manager, args, ctx):
        _require_window(manager, args["window_id"])
        manager.activate_window(args["window_id"])
        return {}


@_command_set.register
class CloseWindow(_cmd.Command):
    op = "close_window"
    category = "delete"
    summary = "Close the canvas window with the given id."
    arguments = {"window_id": _int("Id of the window to close.")}

    def apply(self, manager, args, ctx):
        _require_window(manager, args["window_id"])
        manager.close_window(args["window_id"])
        return {}


@_command_set.register
class SaveImage(_cmd.Command):
    op = "save_image"
    category = "update"
    summary = "Save a canvas window's image to a file path."
    arguments = {"window_id": _int("Id of the window to save."),
                 "path": {**STRING,
                          "description": "Filesystem path to write the image "
                                         "to; the suffix picks the format."}}

    def apply(self, manager, args, ctx):
        _require_window(manager, args["window_id"])
        # The contract returns None, but a Qt backend delegating to
        # R2DWidget.saveImage() returns False when the write fails; surface
        # that as a clean failure instead of a bogus success.
        if manager.save_image(args["window_id"], args["path"]) is False:
            raise _cmd.CommandError(
                f"could not save window {args['window_id']} "
                f"to {args['path']!r}")
        return {}


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
