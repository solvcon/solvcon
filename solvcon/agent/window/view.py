# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Agent View command family: driving a ``ViewTransform2dFp64`` from an agent.

The commands aim the 2D view by panning, zooming, and mapping between screen
and world coordinates. Screen coordinates are Qt pixels (+Y down); world
coordinates are the ``World``'s native units, +Y up. The affine map is
``screen = zoom * world + pan`` with a +Y flip on the y axis.
"""

from .. import _command as _cmd


# Shared JSON Schema fragments, used by reference; treat as immutable.
NUMBER = {"type": "number"}
POSITIVE = {"type": "number", "exclusiveMinimum": 0}


def _num(description):
    return {**NUMBER, "description": description}


def _pos(description):
    return {**POSITIVE, "description": description}


# Mirrors ViewTransform2dFp64, math convention +Y up.
VIEW = {"type": "object",
        "description": "The 2D view transform: screen pan and zoom.",
        "properties": {
            "pan_x": _num("Screen-pixel x offset added to scaled world x."),
            "pan_y": _num("Screen-pixel y offset; pairs with the +Y flip."),
            "zoom": _pos("Screen pixels per world unit; stays positive.")},
        "required": ["pan_x", "pan_y", "zoom"],
        "additionalProperties": False}


_command_set = _cmd.CommandSet(
    "Agent View command", "Any single command in the Agent View schema.")


@_command_set.register
class GetView(_cmd.Command):
    op = "get_view"
    category = "read"
    summary = "Read the current view transform: pan_x, pan_y, and zoom."
    returns = {"view": VIEW}

    def apply(self, view, args, ctx):
        return {"view": {"pan_x": view.pan_x, "pan_y": view.pan_y,
                         "zoom": view.zoom}}


@_command_set.register
class ScreenFromWorld(_cmd.Command):
    op = "screen_from_world"
    category = "read"
    summary = "Map a world point to its screen pixel under the current view."
    arguments = {"world_x": _num("World x of the point (+Y up)."),
                 "world_y": _num("World y of the point (+Y up).")}
    returns = {"screen_x": _num("Screen x in pixels (+Y down)."),
               "screen_y": _num("Screen y in pixels (+Y down).")}

    def apply(self, view, args, ctx):
        screen_x, screen_y = view.screen_from_world(
            args["world_x"], args["world_y"])
        return {"screen_x": screen_x, "screen_y": screen_y}


@_command_set.register
class WorldFromScreen(_cmd.Command):
    op = "world_from_screen"
    category = "read"
    summary = "Map a screen pixel back to a world point under the view."
    arguments = {"screen_x": _num("Screen x in pixels (+Y down)."),
                 "screen_y": _num("Screen y in pixels (+Y down).")}
    returns = {"world_x": _num("World x of the point (+Y up)."),
               "world_y": _num("World y of the point (+Y up).")}

    def apply(self, view, args, ctx):
        world_x, world_y = view.world_from_screen(
            args["screen_x"], args["screen_y"])
        return {"world_x": world_x, "world_y": world_y}


@_command_set.register
class Pan(_cmd.Command):
    op = "pan"
    category = "update"
    summary = "Translate the view by a screen-pixel delta (dx, dy)."
    arguments = {"dx_screen": _num("Screen-pixel delta along x."),
                 "dy_screen": _num("Screen-pixel delta along y.")}

    def apply(self, view, args, ctx):
        view.pan(args["dx_screen"], args["dy_screen"])
        return {}


@_command_set.register
class ZoomAt(_cmd.Command):
    op = "zoom_at"
    category = "update"
    summary = ("Multiply zoom by factor, holding the world point under the "
               "screen anchor fixed.")
    arguments = {"factor": _pos("Zoom multiplier; >1 zooms in, must be > 0."),
                 "anchor_screen_x": _num("Screen x held fixed during zoom."),
                 "anchor_screen_y": _num("Screen y held fixed during zoom.")}

    def apply(self, view, args, ctx):
        view.zoom_at(args["factor"], args["anchor_screen_x"],
                     args["anchor_screen_y"])
        return {}


@_command_set.register
class ZoomAtClamped(_cmd.Command):
    op = "zoom_at_clamped"
    category = "update"
    summary = ("Anchored zoom that keeps the effective zoom within "
               "[min_zoom, max_zoom].")
    arguments = {"factor": _pos("Zoom multiplier; must be > 0."),
                 "anchor_screen_x": _num("Screen x held fixed during zoom."),
                 "anchor_screen_y": _num("Screen y held fixed during zoom."),
                 "min_zoom": _pos("Lower zoom bound; must be > 0."),
                 "max_zoom": _pos("Upper zoom bound; must be >= min_zoom.")}

    def apply(self, view, args, ctx):
        # The schema bounds each limit above zero but cannot compare the two;
        # without this check inverted bounds no-op in C++ yet report success.
        if args["max_zoom"] < args["min_zoom"]:
            raise _cmd.CommandError("max_zoom must be >= min_zoom")
        view.zoom_at_clamped(args["factor"], args["anchor_screen_x"],
                             args["anchor_screen_y"], args["min_zoom"],
                             args["max_zoom"])
        return {}


@_command_set.register
class SetView(_cmd.Command):
    op = "set_view"
    category = "update"
    summary = "Set the view transform directly from pan_x, pan_y, and zoom."
    arguments = {"pan_x": _num("Screen-pixel x offset."),
                 "pan_y": _num("Screen-pixel y offset."),
                 "zoom": _pos("Screen pixels per world unit; must be > 0.")}

    def apply(self, view, args, ctx):
        view.pan_x = args["pan_x"]
        view.pan_y = args["pan_y"]
        view.zoom = args["zoom"]
        return {}


@_command_set.register
class ResetView(_cmd.Command):
    op = "reset_view"
    category = "update"
    summary = "Reset the view to identity: pan (0, 0) and zoom 1."

    def apply(self, view, args, ctx):
        view.reset()
        return {}


class Executor(_cmd.CommandProcessor):
    """Bind the view command set to a ``ViewTransform2dFp64`` target."""

    def __init__(self, view, validate_results=False, reraise=False):
        super().__init__(view, _command_set,
                         validate_results=validate_results, reraise=reraise)


__all__ = (
    "Executor",
    *_cmd.install_command_api(globals(), _command_set),
)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
